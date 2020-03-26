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

import ai.djl.mxnet.engine.MxNDArray;
import ai.djl.mxnet.engine.MxNDManager;
import ai.djl.mxnet.javacpp.AtomicSymbolCreator;
import ai.djl.mxnet.javacpp.NDArrayHandle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.util.PairList;
import java.util.List;

/** A FunctionInfo represents an operator (ie function) within the MXNet Engine. */
public class FunctionInfo {

    private AtomicSymbolCreator handle;
    private String name;
    private PairList<String, String> arguments;

    FunctionInfo(
            AtomicSymbolCreator pointer, String functionName, PairList<String, String> arguments) {
        this.handle = pointer;
        this.name = functionName;
        this.arguments = arguments;
    }

    /**
     * Calls an operator with the given arguments.
     *
     * @param manager the manager to attach the result to
     * @param src the input NDArray(s) to the operator
     * @param dest the destination NDArray(s) to be overwritten with the result of the operator
     * @param params the non-NDArray arguments to the operator. Should be a {@code PairList<String,
     *     String>}
     */
    public void invoke(
            MxNDManager manager, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        JnaUtils.imperativeInvoke(handle, src, dest, params).size();
    }

    /**
     * Calls an operator with the given arguments.
     *
     * @param manager the manager to attach the result to
     * @param src the input NDArray(s) to the operator
     * @param params the non-NDArray arguments to the operator. Should be a {@code PairList<String,
     *     String>}
     * @return the error code or zero for no errors
     */
    public NDArray[] invoke(MxNDManager manager, NDArray[] src, PairList<String, ?> params) {
        NDArray[] dest = new NDArray[0];

        PairList<NDArrayHandle, SparseFormat> pairList =
                JnaUtils.imperativeInvoke(handle, src, dest, params);
        return pairList.stream()
                .map(
                        pair -> {
                            if (pair.getValue() != SparseFormat.DENSE) {
                                return manager.create(pair.getKey(), pair.getValue());
                            }
                            return manager.create(pair.getKey());
                        })
                .toArray(MxNDArray[]::new);
    }

    /**
     * Returns the name of the operator.
     *
     * @return the name of the operator
     */
    public String getFunctionName() {
        return name;
    }

    /**
     * Returns the names of the params to the operator.
     *
     * @return the names of the params to the operator
     */
    public List<String> getArgumentNames() {
        return arguments.keys();
    }

    /**
     * Returns the types of the operator arguments.
     *
     * @return the types of the operator arguments
     */
    public List<String> getArgumentTypes() {
        return arguments.values();
    }
}
