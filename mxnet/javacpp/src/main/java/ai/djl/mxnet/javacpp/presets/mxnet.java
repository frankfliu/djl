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
package ai.djl.mxnet.javacpp.presets;

import java.util.List;
import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.LoadEnabled;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
        target = "ai.djl.mxnet.javacpp",
        global = "ai.djl.mxnet.javacpp.global.mxnet",
        value = {
            @Platform(
                    value = {"linux", "macosx", "windows"},
                    compiler = {"cpp11", "fastfpu"},
                    define = {
                        "DMLC_USE_CXX11 1",
                        "MSHADOW_USE_CBLAS 1",
                        "MSHADOW_IN_CXX11 1",
                        "MSHADOW_USE_CUDA 0",
                        "MSHADOW_USE_F16C 0",
                        "MXNET_USE_TVM_OP 0"
                    },
                    include = {"mxnet/c_api.h", "nnvm/c_api.h"},
                    link = "mxnet",
                    includepath = {
                        "/System/Library/Frameworks/vecLib.framework/",
                        "/System/Library/Frameworks/Accelerate.framework/"
                    }),
            @Platform(
                    value = {
                        "linux-arm64",
                        "linux-ppc64le",
                        "linux-x86_64",
                        "macosx-x86_64",
                        "windows-x86_64"
                    },
                    define = {
                        "DMLC_USE_CXX11 1",
                        "MSHADOW_USE_CBLAS 1",
                        "MSHADOW_IN_CXX11 1",
                        "MSHADOW_USE_CUDA 1",
                        "MSHADOW_USE_F16C 0",
                        "MXNET_USE_TVM_OP 0"
                    },
                    link = {"mxnet"})
        })
public class mxnet implements LoadEnabled, InfoMapper {

    @Override
    public void init(ClassProperties properties) {
        List<String> preloadpath = properties.get("platform.preloadpath");
        preloadpath.add(System.getProperty("ai.djl.mxnet.native_path"));
    }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("MXNET_EXTERN_C", "MXNET_DLL", "NNVM_DLL").cppTypes().annotations())
                .put(new Info("MSHADOW_USE_F16C", "MXNET_USE_TVM_OP").define(false))
                .put(
                        new Info(
                                        "MXNDArrayCreateSparseEx64",
                                        "MXNDArrayGetAuxNDArray64",
                                        "MXNDArrayGetAuxType64",
                                        "MXNDArrayGetShape64",
                                        "MXSymbolInferShape64",
                                        "MXSymbolInferShapePartial64")
                                .skip())
                .put(
                        new Info("NDArrayHandle")
                                .valueTypes("NDArrayHandle")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"NDArrayHandle*\") @ByPtrPtr NDArrayHandle"))
                .put(
                        new Info("const NDArrayHandle")
                                .valueTypes("NDArrayHandle")
                                .pointerTypes(
                                        "@Cast(\"NDArrayHandle*\") PointerPointer",
                                        "@Cast(\"NDArrayHandle*\") @ByPtrPtr NDArrayHandle"))
                .put(
                        new Info("FunctionHandle")
                                .annotations("@Const")
                                .valueTypes("FunctionHandle")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"FunctionHandle*\") @ByPtrPtr FunctionHandle"))
                .put(
                        new Info("AtomicSymbolCreator")
                                .valueTypes("AtomicSymbolCreator")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"AtomicSymbolCreator*\") @ByPtrPtr AtomicSymbolCreator"))
                .put(
                        new Info("SymbolHandle")
                                .valueTypes("SymbolHandle")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"SymbolHandle*\") @ByPtrPtr SymbolHandle"))
                .put(
                        new Info("const SymbolHandle")
                                .valueTypes("SymbolHandle")
                                .pointerTypes(
                                        "@Cast(\"SymbolHandle*\") PointerPointer",
                                        "@Cast(\"SymbolHandle*\") @ByPtrPtr SymbolHandle"))
                .put(
                        new Info("AtomicSymbolHandle")
                                .valueTypes("AtomicSymbolHandle")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"AtomicSymbolHandle*\") @ByPtrPtr AtomicSymbolHandle"))
                .put(
                        new Info("ExecutorHandle")
                                .valueTypes("ExecutorHandle")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"ExecutorHandle*\") @ByPtrPtr ExecutorHandle"))
                .put(
                        new Info("DataIterCreator")
                                .valueTypes("DataIterCreator")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"DataIterCreator*\") @ByPtrPtr DataIterCreator"))
                .put(
                        new Info("DataIterHandle")
                                .valueTypes("DataIterHandle")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"DataIterHandle*\") @ByPtrPtr DataIterHandle"))
                .put(
                        new Info("KVStoreHandle")
                                .valueTypes("KVStoreHandle")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"KVStoreHandle*\") @ByPtrPtr KVStoreHandle"))
                .put(
                        new Info("RecordIOHandle")
                                .valueTypes("RecordIOHandle")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"RecordIOHandle*\") @ByPtrPtr RecordIOHandle"))
                .put(
                        new Info("RtcHandle")
                                .valueTypes("RtcHandle")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"RtcHandle*\") @ByPtrPtr RtcHandle"))
                .put(
                        new Info("OptimizerCreator")
                                .valueTypes("OptimizerCreator")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"OptimizerCreator*\") @ByPtrPtr OptimizerCreator"))
                .put(
                        new Info("OptimizerHandle")
                                .valueTypes("OptimizerHandle")
                                .pointerTypes(
                                        "PointerPointer",
                                        "@Cast(\"OptimizerHandle*\") @ByPtrPtr OptimizerHandle"));
    }
}
