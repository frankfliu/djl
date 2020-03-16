/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.mxnet.jna;

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.mxnet.engine.CachedOp;
import ai.djl.mxnet.engine.MxDeviceType;
import ai.djl.mxnet.engine.MxNDArray;
import ai.djl.mxnet.engine.MxNDManager;
import ai.djl.mxnet.engine.MxSymbolBlock;
import ai.djl.mxnet.engine.Symbol;
import ai.djl.mxnet.javacpp.AtomicSymbolCreator;
import ai.djl.mxnet.javacpp.CachedOpHandle;
import ai.djl.mxnet.javacpp.KVStoreHandle;
import ai.djl.mxnet.javacpp.LibFeature;
import ai.djl.mxnet.javacpp.MXKVStoreStrUpdater;
import ai.djl.mxnet.javacpp.MXKVStoreUpdater;
import ai.djl.mxnet.javacpp.NDArrayHandle;
import ai.djl.mxnet.javacpp.OpHandle;
import ai.djl.mxnet.javacpp.SymbolHandle;
import ai.djl.mxnet.javacpp.global.mxnet;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.Parameter;
import ai.djl.util.PairList;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class containing utilities to interact with the MXNet Engine's Java Native Access (JNA) layer.
 */
@SuppressWarnings("MissingJavadocMethod")
public final class JnaUtils {

    private static final Logger logger = LoggerFactory.getLogger(JnaUtils.class);

    public static final String[] EMPTY_ARRAY = new String[0];

    /** An enum that enumerates the statuses of numpy mode. */
    public enum NumpyMode {
        OFF,
        THREAD_LOCAL_ON,
        GLOBAL_ON
    }

    private static final String[] OP_NAME_PREFIX = {
        "_contrib_", "_linalg_", "_sparse_", "_image_", "_random_"
    };

    public static final String MXNET_THREAD_SAFE_PREDICTOR = "MXNET_THREAD_SAFE_PREDICTOR";

    static {
        Path path = Paths.get(LibUtils.getLibName());
        logger.debug("Loading mxnet library from: {}", path);
        if (path.isAbsolute()) {
            Path cacheDir = path.getParent().toAbsolutePath();
            System.setProperty("ai.djl.mxnet.native_path", cacheDir.toString());
            LibUtils.copyJniFromClasspath(cacheDir);
        }
    }

    private static final Map<String, FunctionInfo> OPS = getNdArrayFunctions();

    private JnaUtils() {}

    /////////////////////////////////
    // MXNet information
    /////////////////////////////////

    public static int getVersion() {
        int[] version = new int[1];
        mxnet.MXGetVersion(version);
        return version[0];
    }

    public static Set<String> getAllOpNames() {
        IntBuffer outSize = IntBuffer.allocate(1);
        PointerPointer<BytePointer> outArray = new PointerPointer<>();

        checkCall(mxnet.MXListAllOpNames(outSize, outArray));

        int size = outSize.get();
        Set<String> set = new HashSet<>();
        for (int i = 0; i < size; ++i) {
            set.add(getStringValue(outArray, i));
        }
        return set;
    }

    public static Map<String, FunctionInfo> getNdArrayFunctions() {
        Set<String> opNames = JnaUtils.getAllOpNames();
        Map<String, FunctionInfo> map = new ConcurrentHashMap<>();

        for (String opName : opNames) {
            OpHandle opHandle = new OpHandle();
            checkCall(mxnet.NNGetOpHandle(opName, opHandle));

            String functionName = getOpNamePrefix(opName);

            // System.out.println("Name: " + opName + "/" + functionName);
            map.put(functionName, getFunctionByName(opName, functionName, opHandle));
        }
        return map;
    }

    public static FunctionInfo op(String opName) {
        if (!OPS.containsKey(opName)) {
            throw new IllegalArgumentException("Unknown operator: " + opName);
        }
        return OPS.get(opName);
    }

    private static FunctionInfo getFunctionByName(
            String name, String functionName, OpHandle handle) {
        PointerPointer<BytePointer> nameRef = new PointerPointer<>(1);
        nameRef.putString(name);
        PointerPointer<BytePointer> description = new PointerPointer<>(1);
        IntPointer numArgs = new IntPointer(1);
        PointerPointer<BytePointer> argNameRef = new PointerPointer<>(1);
        PointerPointer<BytePointer> argTypeRef = new PointerPointer<>(1);
        PointerPointer<BytePointer> argDescRef = new PointerPointer<>(1);
        PointerPointer<BytePointer> keyVarArgs = new PointerPointer<>(1);
        PointerPointer<BytePointer> returnType = new PointerPointer<>(1);

        AtomicSymbolCreator symbolCreator = new AtomicSymbolCreator(handle);

        checkCall(
                mxnet.MXSymbolGetAtomicSymbolInfo(
                        symbolCreator,
                        nameRef,
                        description,
                        numArgs,
                        argNameRef,
                        argTypeRef,
                        argDescRef,
                        keyVarArgs,
                        returnType));

        int count = numArgs.get();
        PairList<String, String> arguments = new PairList<>(count);
        if (count != 0) {
            for (int i = 0; i < count; ++i) {
                String argName = getStringValue(argNameRef, i);
                String argType = getStringValue(argTypeRef, i);
                arguments.add(argName, argType);
            }
        }

        return new FunctionInfo(symbolCreator, functionName, arguments);
    }

    /////////////////////////////////
    // System information
    /////////////////////////////////

    public static int getGpuCount() {
        IntBuffer count = IntBuffer.allocate(1);
        checkCall(mxnet.MXGetGPUCount(count));

        return count.get();
    }

    public static long[] getGpuMemory(Device device) {
        if (!Device.Type.GPU.equals(device.getDeviceType())) {
            throw new IllegalArgumentException("Only GPU device is allowed.");
        }

        int deviceId = device.getDeviceId();
        long[] ret = new long[2];

        LongBuffer freeMem = LongBuffer.wrap(ret, 0, 1);
        LongBuffer totalMem = LongBuffer.wrap(ret, 1, 1);

        checkCall(mxnet.MXGetGPUMemoryInformation64(deviceId, freeMem, totalMem));

        return ret;
    }

    /////////////////////////////////
    // Utilities
    /////////////////////////////////

    public static Set<String> getFeatures() {
        PointerPointer<LibFeature> features = new PointerPointer<>(1);
        SizeTPointer outSize = new SizeTPointer(1);
        checkCall(mxnet.MXLibInfoFeatures(features, outSize));

        long size = outSize.get(0);
        if (size == 0) {
            return Collections.emptySet();
        }

        Pointer p = features.get();
        Set<String> set = new HashSet<>();
        for (int i = 0; i < size; ++i) {
            LibFeature feature = new LibFeature(p);
            if (feature.enabled()) {
                set.add(getStringValue(feature.name()));
            }
            p = new MyPointer(p, feature.sizeof());
        }
        return set;
    }

    public static int randomSeed(int seed) {
        return mxnet.MXRandomSeed(seed);
    }

    /////////////////////////////////
    // NDArray
    /////////////////////////////////

    public static NDArrayHandle createNdArray(
            Device device, Shape shape, DataType dtype, int size, boolean delayedAlloc) {
        int deviceType = MxDeviceType.toDeviceType(device);
        int deviceId = (deviceType != 1) ? device.getDeviceId() : -1;
        int delay = delayedAlloc ? 1 : 0;

        NDArrayHandle handle = new NDArrayHandle();
        int[] shapeArray = Arrays.stream(shape.getShape()).mapToInt(Math::toIntExact).toArray();
        checkCall(
                mxnet.MXNDArrayCreateEx(
                        shapeArray, size, deviceType, deviceId, delay, dtype.ordinal(), handle));

        return handle;
    }

    public static NDArrayHandle createSparseNdArray(
            SparseFormat fmt,
            Device device,
            Shape shape,
            DataType dtype,
            DataType[] auxDTypes,
            Shape[] auxShapes,
            boolean delayedAlloc) {
        int[] shapeArray = Arrays.stream(shape.getShape()).mapToInt(Math::toIntExact).toArray();
        int deviceType = MxDeviceType.toDeviceType(device);
        int deviceId = (deviceType != 1) ? device.getDeviceId() : -1;
        int delay = delayedAlloc ? 1 : 0;
        NDArrayHandle handle = new NDArrayHandle();
        int[] auxDTypesInt = Arrays.stream(auxDTypes).mapToInt(DataType::ordinal).toArray();
        int[] auxNDims = Arrays.stream(auxShapes).mapToInt(Shape::dimension).toArray();
        int[] auxShapesInt = Arrays.stream(auxShapes).mapToInt(ele -> (int) ele.head()).toArray();
        checkCall(
                mxnet.MXNDArrayCreateSparseEx(
                        fmt.getValue(),
                        shapeArray,
                        shapeArray.length,
                        deviceType,
                        deviceId,
                        delay,
                        dtype.ordinal(),
                        auxDTypes.length,
                        auxDTypesInt,
                        auxNDims,
                        auxShapesInt,
                        handle));
        return handle;
    }

    public static void ndArraySyncCopyFromNdArray(MxNDArray dest, MxNDArray src, int location) {
        checkCall(mxnet.MXNDArraySyncCopyFromNDArray(dest.getHandle(), src.getHandle(), location));
    }

    public static NDList loadNdArray(MxNDManager manager, Path path, Device device) {
        IntBuffer handlesSize = IntBuffer.allocate(1);
        PointerPointer<NDArrayHandle> handlesRef = new PointerPointer<>(1);
        PointerPointer<BytePointer> namesRef = new PointerPointer<>();
        IntBuffer namesSize = IntBuffer.allocate(1);
        checkCall(
                mxnet.MXNDArrayLoad(path.toString(), handlesSize, handlesRef, namesSize, namesRef));
        int ndArrayCount = handlesSize.get();
        int nameCount = namesSize.get();
        if (nameCount > 0 && ndArrayCount != nameCount) {
            throw new IllegalStateException(
                    "Mismatch between names and arrays in checkpoint file: " + path.toString());
        }
        NDList ndList = new NDList();

        for (int i = 0; i < ndArrayCount; i++) {
            NDArrayHandle handle = handlesRef.get(NDArrayHandle.class, i);
            NDArray array = manager.create(handle);
            array.getShape();
            if (nameCount > 0) {
                array.setName(getStringValue(namesRef, i));
            }
            ndList.add(array);
        }

        // MXNet always load NDArray on CPU
        if (Device.cpu().equals(device)) {
            return ndList;
        }

        NDList ret = ndList.asInDevice(device, true);
        ndList.close();
        return ret;
    }

    public static void freeNdArray(NDArrayHandle handle) {
        checkNullHandle(handle, "free");
        checkCall(mxnet.MXNDArrayFree(handle));
    }

    public static void waitToRead(NDArrayHandle handle) {
        checkNullHandle(handle, "wait to read");
        checkCall(mxnet.MXNDArrayWaitToRead(handle));
    }

    public static void waitToWrite(NDArrayHandle handle) {
        checkNullHandle(handle, "wait to write");
        checkCall(mxnet.MXNDArrayWaitToWrite(handle));
    }

    public static void waitAll() {
        checkCall(mxnet.MXNDArrayWaitAll());
    }

    public static void syncCopyToCPU(NDArrayHandle handle, ByteBuffer data, int len) {
        checkNullHandle(handle, "copy from");
        BytePointer pointer = new BytePointer(data);
        checkCall(mxnet.MXNDArraySyncCopyToCPU(handle, pointer, len));
    }

    public static void syncCopyFromCPU(NDArrayHandle handle, Pointer data, long len) {
        checkCall(mxnet.MXNDArraySyncCopyFromCPU(handle, data, len));
    }

    public static PairList<NDArrayHandle, SparseFormat> imperativeInvoke(
            AtomicSymbolCreator function,
            NDArray[] src,
            NDArray[] dest,
            PairList<String, ?> params) {
        PointerPointer<BytePointer> keys;
        PointerPointer<BytePointer> values;
        int size;
        if (params == null) {
            size = 0;
            keys = new PointerPointer<>(1);
            keys.put(new BytePointer());
            values = new PointerPointer<>(1);
            values.put(new BytePointer());
        } else {
            size = params.size();
            keys = toPointerArray(params.keys());
            String[] vals = params.values().stream().map(Object::toString).toArray(String[]::new);
            values = toPointerArray(vals);
        }
        PointerPointer<NDArrayHandle> srcPointer = toPointerArray(src);
        PointerPointer<NDArrayHandle> destPointer = toPointerArray(dest);

        PointerPointer<IntPointer> destSType = new PointerPointer<>(1);
        destSType.put(new IntPointer());
        IntPointer numOutputs = new IntPointer(1);
        numOutputs.put(1);

        checkCall(
                mxnet.MXImperativeInvokeEx(
                        function,
                        src.length,
                        srcPointer,
                        numOutputs,
                        destPointer,
                        size,
                        keys,
                        values,
                        destSType));
        int numOfOutputs = numOutputs.get(0);
        PairList<NDArrayHandle, SparseFormat> pairList = new PairList<>();

        if (dest.length == 0) {
            Pointer p = destPointer.get();
            for (int i = 0; i < numOfOutputs; ++i) {
                NDArrayHandle handle = new NDArrayHandle(p);
                p = new MyPointer(p, handle.sizeof());

                int fmt = destSType.get(IntPointer.class, i).get();
                pairList.add(handle, SparseFormat.fromValue(fmt));
            }
        } else {
            PointerPointer<NDArrayHandle> list = new PointerPointer<>(destPointer.get());
            for (int i = 0; i < numOfOutputs; i++) {
                NDArrayHandle handle = list.get(NDArrayHandle.class, i);
                int fmt = destSType.get(IntPointer.class, i).get();
                pairList.add(handle, SparseFormat.fromValue(fmt));
            }
        }
        return pairList;
    }

    public static SparseFormat getStorageType(NDArrayHandle ndArray) {
        IntBuffer type = IntBuffer.allocate(1);
        checkNullHandle(ndArray, "get the storage type of");
        checkCall(mxnet.MXNDArrayGetStorageType(ndArray, type));
        return SparseFormat.fromValue(type.get());
    }

    public static Device getDevice(NDArrayHandle ndArray) {
        IntBuffer deviceType = IntBuffer.allocate(1);
        IntBuffer deviceId = IntBuffer.allocate(1);
        checkNullHandle(ndArray, "get the device of");
        checkCall(mxnet.MXNDArrayGetContext(ndArray, deviceType, deviceId));
        String deviceTypeStr = MxDeviceType.fromDeviceType(deviceType.get(0));
        // CPU is special case which don't have device id
        if (Device.Type.CPU.equals(deviceTypeStr)) {
            return new Device(Device.Type.CPU);
        }
        return new Device(deviceTypeStr, deviceId.get(0));
    }

    public static Shape getShape(NDArrayHandle ndArray) {
        IntPointer dim = new IntPointer(1);
        LongPointer out = new LongPointer();
        checkNullHandle(ndArray, "get the shape of");
        checkCall(mxnet.MXNDArrayGetShapeEx64(ndArray, dim, out));
        int nDim = dim.get(0);
        if (nDim == 0) {
            return new Shape();
        }
        long[] shape = new long[nDim];
        for (int i = 0; i < nDim; ++i) {
            shape[i] = out.get(i);
        }
        return new Shape(shape);
    }

    public static DataType getDataType(NDArrayHandle ndArray) {
        IntBuffer dataType = IntBuffer.allocate(1);
        checkNullHandle(ndArray, "get the data type of");
        checkCall(mxnet.MXNDArrayGetDType(ndArray, dataType));
        return DataType.values()[dataType.get()];
    }

    /////////////////////////////////
    // MxGradientCollector
    /////////////////////////////////
    public static boolean autogradSetIsRecording(boolean isRecording) {
        IntBuffer prev = IntBuffer.allocate(1);
        checkCall(mxnet.MXAutogradSetIsRecording(isRecording ? 1 : 0, prev));
        return prev.get(0) == 1;
    }

    public static boolean autogradSetTraining(boolean isTraining) {
        IntBuffer prev = IntBuffer.allocate(1);
        checkCall(mxnet.MXAutogradSetIsTraining(isTraining ? 1 : 0, prev));
        return prev.get(0) == 1;
    }

    public static boolean autogradIsRecording() {
        boolean[] isRecording = new boolean[1];
        checkCall(mxnet.MXAutogradIsRecording(isRecording));
        return isRecording[0];
    }

    public static boolean autogradIsTraining() {
        boolean[] isTraining = new boolean[1];
        checkCall(mxnet.MXAutogradIsTraining(isTraining));
        return isTraining[0];
    }

    public static void autogradMarkVariables(
            int numVar, NDArrayHandle varHandles, IntBuffer reqsArray, NDArrayHandle gradHandle) {
        PointerPointer<NDArrayHandle> varRef = new PointerPointer<>(varHandles);
        PointerPointer<NDArrayHandle> gradRef = new PointerPointer<>(gradHandle);
        checkCall(mxnet.MXAutogradMarkVariables(numVar, varRef, reqsArray, gradRef));
    }

    public static void autogradBackward(NDList array, int retainGraph) {
        checkCall(
                mxnet.MXAutogradBackward(
                        array.size(),
                        toPointerArray(array),
                        new PointerPointer<BytePointer>(),
                        retainGraph));
    }

    public static void autogradBackwardExecute(
            int numOutput,
            NDList array,
            NDArray outgrad,
            int numVariables,
            NDArrayHandle varHandles,
            int retainGraph,
            int createGraph,
            int isTrain,
            NDArrayHandle gradHandles,
            PointerPointer<IntPointer> gradSparseFormat) {
        checkCall(
                mxnet.MXAutogradBackwardEx(
                        numOutput,
                        toPointerArray(array),
                        new PointerPointer<NDArrayHandle>(),
                        numVariables,
                        new PointerPointer<NDArrayHandle>(varHandles),
                        retainGraph,
                        createGraph,
                        isTrain,
                        new PointerPointer<>(gradHandles),
                        gradSparseFormat));
    }

    public static SymbolHandle autogradGetSymbol(NDArray array) {
        NDArrayHandle handle = ((MxNDArray) array).getHandle();
        PointerPointer<SymbolHandle> out = new PointerPointer<>();
        checkCall(mxnet.MXAutogradGetSymbol(handle, out));
        return out.get(SymbolHandle.class, 0);
    }

    public static int isNumpyMode() {
        IntBuffer ret = IntBuffer.allocate(1);
        checkCall(mxnet.MXIsNumpyShape(ret));
        return ret.get();
    }

    public static void setNumpyMode(JnaUtils.NumpyMode mode) {
        IntBuffer ret = IntBuffer.allocate(1);
        checkCall(mxnet.MXSetIsNumpyShape(mode.ordinal(), ret));
    }

    public static NDArrayHandle getGradient(NDArrayHandle handle) {
        PointerPointer<NDArrayHandle> ref = new PointerPointer<>();
        checkNullHandle(handle, "get the gradient for");
        checkCall(mxnet.MXNDArrayGetGrad(handle, ref));
        return ref.get(NDArrayHandle.class, 0);
    }

    public static KVStoreHandle parameterStoreCreate(String type) {
        KVStoreHandle handle = new KVStoreHandle();
        checkCall(mxnet.MXKVStoreCreate(type, handle));
        return handle;
    }

    public static void parameterStoreClose(KVStoreHandle handle) {
        checkCall(mxnet.MXKVStoreFree(handle));
    }

    public static void parameterStoreInit(
            KVStoreHandle handle, int num, String[] keys, NDList ndList) {
        checkNullHandle(handle, "initialize the parameter store with");
        PointerPointer<BytePointer> keyPointers = toPointerArray(keys);
        checkCall(mxnet.MXKVStoreInitEx(handle, num, keyPointers, toPointerArray(ndList)));
    }

    public static void parameterStorePush(
            KVStoreHandle handle, int num, String[] keys, NDList ndList, int priority) {
        checkNullHandle(handle, "push to the parameter store with");
        PointerPointer<BytePointer> keyPointers = toPointerArray(keys);
        PointerPointer<NDArrayHandle> arrayPointers = toPointerArray(ndList);
        checkCall(mxnet.MXKVStorePushEx(handle, num, keyPointers, arrayPointers, priority));
    }

    public static void parameterStorePull(
            KVStoreHandle handle, int num, int[] keys, NDList vals, int priority) {
        checkNullHandle(handle, "pull from the parameter store with");
        checkCall(mxnet.MXKVStorePull(handle, num, keys, toPointerArray(vals), priority));
    }

    public static void parameterStorePull(
            KVStoreHandle handle, int num, String[] keys, NDList ndList, int priority) {
        checkNullHandle(handle, "pull from the parameter store with");
        PointerPointer<BytePointer> keyPointers = toPointerArray(keys);
        PointerPointer<NDArrayHandle> arrayPointers = toPointerArray(ndList);
        checkCall(mxnet.MXKVStorePullEx(handle, num, keyPointers, arrayPointers, priority));
    }

    public static void parameterStoreSetUpdater(
            KVStoreHandle handle,
            MXKVStoreUpdater updater,
            MXKVStoreStrUpdater stringUpdater,
            Pointer updaterHandle) {
        checkCall(mxnet.MXKVStoreSetUpdaterEx(handle, updater, stringUpdater, updaterHandle));
    }

    public static void parameterStoreSetUpdater(
            KVStoreHandle handle, MXKVStoreUpdater updater, Pointer updaterHandle) {
        checkCall(mxnet.MXKVStoreSetUpdater(handle, updater, updaterHandle));
    }

    /////////////////////////////////
    // MXNet Symbols
    /////////////////////////////////

    public static SymbolHandle getSymbolOutput(SymbolHandle symbol, int index) {
        PointerPointer<SymbolHandle> out = new PointerPointer<>();
        checkCall(mxnet.MXSymbolGetOutput(symbol, index, out));
        return out.get(SymbolHandle.class, 0);
    }

    public static String[] listSymbolOutputs(SymbolHandle symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerPointer<BytePointer> ref = new PointerPointer<>();

        checkCall(mxnet.MXSymbolListOutputs(symbol, size, ref));
        return toStringArray(ref, size.get());
    }

    public static void freeSymbol(SymbolHandle symbol) {
        checkCall(mxnet.MXSymbolFree(symbol));
    }

    public static String[] listSymbolNames(SymbolHandle symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerPointer<BytePointer> ref = new PointerPointer<>();

        checkCall(mxnet.NNSymbolListInputNames(symbol, 0, size, ref));

        return toStringArray(ref, size.get());
    }

    public static String[] listSymbolArguments(SymbolHandle symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerPointer<BytePointer> ref = new PointerPointer<>();

        checkCall(mxnet.MXSymbolListArguments(symbol, size, ref));

        return toStringArray(ref, size.get());
    }

    public static String[] listSymbolAuxiliaryStates(SymbolHandle symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerPointer<BytePointer> ref = new PointerPointer<>();

        checkCall(mxnet.MXSymbolListAuxiliaryStates(symbol, size, ref));

        return toStringArray(ref, size.get());
    }

    public static SymbolHandle getSymbolInternals(SymbolHandle symbol) {
        PointerPointer<SymbolHandle> handle = new PointerPointer<>();
        checkCall(mxnet.MXSymbolGetInternals(symbol, handle));
        return handle.get(SymbolHandle.class, 0);
    }

    public static SymbolHandle createSymbolFromFile(String path) {
        SymbolHandle handle = new SymbolHandle();
        checkCall(mxnet.MXSymbolCreateFromFile(path, handle));
        return handle;
    }

    private static List<Shape> recoverShape(
            long size, PointerPointer<IntPointer> nDim, PointerPointer<LongPointer> data) {
        if (size == 0) {
            return new ArrayList<>();
        }
        List<Shape> result = new ArrayList<>((int) size);
        int idx = 0;
        for (int i = 0; i < size; ++i) {
            int dim = nDim.get(IntPointer.class, i).get(0);
            long[] shape = new long[dim];
            for (int j = 0; j < dim; ++j) {
                shape[j] = data.get(LongPointer.class, idx + j).get(0);
            }
            idx += dim;
            result.add(new Shape(shape));
        }
        return result;
    }

    public static List<List<Shape>> inferShape(Symbol symbol, PairList<String, Shape> args) {
        SymbolHandle handle = symbol.getHandle();
        int numArgs = args.size();
        PointerPointer<BytePointer> keys = toPointerArray(args.keys());
        // the following two is also the representation of
        // CSR NDArray
        LongPointer indPtr = new LongPointer(numArgs + 1);
        Shape flattened = new Shape();
        indPtr.put(0, 0);
        for (int i = 0; i < args.size(); i++) {
            Shape shape = args.valueAt(i);
            indPtr.put(shape.dimension(), i + 1);
            flattened = flattened.addAll(shape);
        }
        long[] shape = flattened.getShape();
        LongPointer flattenedShapeArray = new LongPointer(shape.length);
        flattenedShapeArray.put(shape);

        SizeTPointer inShapeSize = new SizeTPointer();
        PointerPointer<IntPointer> inShapeNDim = new PointerPointer<>();
        PointerPointer<LongPointer> inShapeData = new PointerPointer<>();
        SizeTPointer outShapeSize = new SizeTPointer();
        PointerPointer<IntPointer> outShapeNDim = new PointerPointer<>();
        PointerPointer<LongPointer> outShapeData = new PointerPointer<>();
        SizeTPointer auxShapeSize = new SizeTPointer();
        PointerPointer<IntPointer> auxShapeNDim = new PointerPointer<>();
        PointerPointer<LongPointer> auxShapeData = new PointerPointer<>();
        IntPointer complete = new IntPointer(1);
        checkCall(
                mxnet.MXSymbolInferShapeEx64(
                        handle,
                        numArgs,
                        keys,
                        indPtr,
                        flattenedShapeArray,
                        inShapeSize,
                        inShapeNDim,
                        inShapeData,
                        outShapeSize,
                        outShapeNDim,
                        outShapeData,
                        auxShapeSize,
                        auxShapeNDim,
                        auxShapeData,
                        complete));

        if (complete.get() != 0) {
            return Arrays.asList(
                    recoverShape(inShapeSize.get(), inShapeNDim, inShapeData),
                    recoverShape(outShapeSize.get(), outShapeNDim, outShapeData),
                    recoverShape(auxShapeSize.get(), auxShapeNDim, auxShapeData));
        }
        return null;
    }

    //////////////////////////////////
    // cached Op
    //////////////////////////////////

    /**
     * Creates cached op flags.
     *
     * <p>data_indices : [0, 2, 4] Used to label input location, param_indices : [1, 3] Used to
     * label param location
     *
     * @param block the {@link MxSymbolBlock} that loaded in the backend
     * @param manager the NDManager used to create NDArray
     * @return a CachedOp for inference
     */
    public static CachedOp createCachedOp(MxSymbolBlock block, MxNDManager manager) {
        Symbol symbol = block.getSymbol();

        List<Parameter> parameters = block.getAllParameters();

        // record data index in all inputs
        PairList<String, Integer> dataIndices = new PairList<>();
        // record parameter index in all inputs
        List<Integer> paramIndices = new ArrayList<>();
        int index = 0;
        for (Parameter parameter : parameters) {
            // We assume uninitialized parameters are data inputs
            if (parameter.isInitialized()) {
                paramIndices.add(index);
            } else {
                dataIndices.add(parameter.getName(), index);
            }
            ++index;
        }

        // Creating CachedOp
        SymbolHandle symbolHandle = symbol.getHandle();
        CachedOpHandle cachedOp = new CachedOpHandle();
        if (useThreadSafePredictor()) {
            PointerPointer<BytePointer> keys = new PointerPointer<>(2);
            PointerPointer<BytePointer> values = new PointerPointer<>(2);
            keys.putString("data_indices", "param_indices");
            values.putString(dataIndices.values().toString(), paramIndices.toString());
            checkCall(mxnet.MXCreateCachedOpEX(symbolHandle, 2, keys, values, cachedOp, true));
        } else {
            // static_alloc and static_shape are enabled by default
            PointerPointer<BytePointer> keys = new PointerPointer<>(4);
            PointerPointer<BytePointer> values = new PointerPointer<>(4);
            keys.putString("data_indices", "param_indices", "static_alloc", "static_shape");
            values.putString(dataIndices.values().toString(), paramIndices.toString(), "1", "1");
            checkCall(mxnet.MXCreateCachedOpEx(symbolHandle, 4, keys, values, cachedOp));
        }

        return new CachedOp(cachedOp, manager, parameters, paramIndices, dataIndices);
    }

    public static void freeCachedOp(CachedOpHandle handle) {
        if (useThreadSafePredictor()) {
            checkCall(mxnet.MXFreeCachedOpEX(handle, true));
        } else {
            checkCall(mxnet.MXFreeCachedOp(handle));
        }
    }

    public static MxNDArray[] cachedOpInvoke(
            MxNDManager manager, CachedOpHandle cachedOpHandle, MxNDArray[] inputs) {
        PointerPointer<NDArrayHandle> inputArrays = toPointerArray(inputs);
        IntPointer buf = new IntPointer(1);
        PointerPointer<NDArrayHandle> outputArrays = new PointerPointer<>();
        PointerPointer<IntPointer> outSTypeRef = new PointerPointer<>(1);
        checkCall(
                mxnet.MXInvokeCachedOpEx(
                        cachedOpHandle,
                        inputs.length,
                        inputArrays,
                        buf,
                        outputArrays,
                        outSTypeRef));

        int numOutputs = buf.get();
        MxNDArray[] output = new MxNDArray[numOutputs];
        IntPointer intPointer = outSTypeRef.get(IntPointer.class, 0);
        for (int i = 0; i < numOutputs; i++) {
            int sType = intPointer.get(i);
            NDArrayHandle handle = outputArrays.get(NDArrayHandle.class, i);
            if (sType != 0) {
                output[i] = manager.create(handle, SparseFormat.fromValue(sType));
            } else {
                output[i] = manager.create(handle);
            }
        }
        return output;
    }

    public static boolean useThreadSafePredictor() {
        return Boolean.getBoolean(MXNET_THREAD_SAFE_PREDICTOR);
    }

    public static void checkCall(int ret) {
        if (ret != 0) {
            throw new EngineException("MXNet engine call failed: " + getLastError());
        }
    }

    private static String getLastError() {
        try (BytePointer p = mxnet.MXGetLastError()) {
            return getStringValue(p);
        }
    }

    private static void checkNullHandle(Pointer handle, String msg) {
        if (handle == null) {
            throw new IllegalArgumentException(
                    "Tried to " + msg + " an MXNet NDArray that was already closed");
        }
    }

    static PointerPointer<NDArrayHandle> toPointerArray(NDList ndList) {
        int size = ndList.size();
        PointerPointer<NDArrayHandle> pointers = new PointerPointer<>(size);
        for (int i = 0; i < size; i++) {
            pointers.put(i, ((MxNDArray) ndList.get(i)).getHandle());
        }
        return pointers;
    }

    static PointerPointer<NDArrayHandle> toPointerArray(NDArray[] arrays) {
        if (arrays.length == 0) {
            return new PointerPointer<>();
        }

        PointerPointer<NDArrayHandle> pointers = new PointerPointer<>(arrays.length);
        for (int i = 0; i < arrays.length; i++) {
            pointers.put(i, ((MxNDArray) arrays[i]).getHandle());
        }
        return pointers;
    }

    private static PointerPointer<BytePointer> toPointerArray(String[] strings) {
        try {
            return new PointerPointer<>(strings, StandardCharsets.UTF_8.name());
        } catch (UnsupportedEncodingException e) {
            throw new AssertionError(e);
        }
    }

    private static PointerPointer<BytePointer> toPointerArray(List<String> list) {
        return toPointerArray(list.toArray(EMPTY_ARRAY));
    }

    private static String[] toStringArray(PointerPointer<BytePointer> pointers, int size) {
        if (size == 0) {
            return new String[0];
        }

        String[] arr = new String[size];
        for (int i = 0; i < size; ++i) {
            arr[i] = getStringValue(pointers, i);
        }
        return arr;
    }

    private static NDArray[] toNDArrays(
            MxNDManager manager, PointerPointer<NDArrayHandle> pointers, int size) {
        if (size == 0) {
            return new NDArray[0];
        }

        NDArray[] arr = new NDArray[size];
        for (int i = 0; i < size; ++i) {
            arr[i] = manager.create(pointers.get(NDArrayHandle.class, i));
        }
        return arr;
    }

    public static String getStringValue(BytePointer pointer) {
        try {
            return pointer.getString(StandardCharsets.UTF_8.name());
        } catch (UnsupportedEncodingException e) {
            throw new AssertionError(e);
        }
    }

    public static String getStringValue(PointerPointer<BytePointer> pointers, long index) {
        try {
            return pointers.getString(index, StandardCharsets.UTF_8.name());
        } catch (UnsupportedEncodingException e) {
            throw new AssertionError(e);
        }
    }

    private static String getOpNamePrefix(String name) {
        for (String prefix : OP_NAME_PREFIX) {
            if (name.startsWith(prefix)) {
                return name.substring(prefix.length());
            }
        }
        return name;
    }

    private static class MyPointer extends Pointer {

        public MyPointer(Pointer p, long offset) {
            super(p);
            address += offset;
        }
    }
}
