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

package ai.djl.mxnet.engine;

import ai.djl.mxnet.javacpp.KVStoreHandle;
import ai.djl.mxnet.javacpp.MXKVStoreStrUpdater;
import ai.djl.mxnet.javacpp.NDArrayHandle;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.mxnet.jna.NativeResource;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterServer;
import ai.djl.training.optimizer.Optimizer;
import java.util.Arrays;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;

/** {@code MxParameterServer} is the MXNet implementation of {@link ParameterServer}. */
public class MxParameterServer extends NativeResource implements ParameterServer {

    /**
     * Constructs a new {@code MxParameterServer}.
     *
     * @param optimizer the optimizer to use for the parameter server updates
     */
    public MxParameterServer(Optimizer optimizer) {
        super(createdKVStore());
        JnaUtils.parameterStoreSetUpdater(
                getHandle(), null, new OptimizerCallback(optimizer), null);
    }

    /** {@inheritDoc} */
    @Override
    public void init(String parameterId, NDArray[] values) {
        String[] keys = new String[values.length];
        Arrays.fill(keys, parameterId);
        NDList ndList = new NDList(values);
        JnaUtils.parameterStoreInit(getHandle(), values.length, keys, ndList);
    }

    /** {@inheritDoc} */
    @Override
    public void push(String parameterId, NDArray[] grads, int priority) {
        String[] keys = new String[grads.length];
        Arrays.fill(keys, parameterId);
        NDList vals = new NDList(grads);
        JnaUtils.parameterStorePush(getHandle(), grads.length, keys, vals, priority);
    }

    /** {@inheritDoc} */
    @Override
    public void pull(String parameterId, NDArray[] weights, int priority) {
        String[] keys = new String[weights.length];
        Arrays.fill(keys, parameterId);
        NDList vals = new NDList(weights);
        JnaUtils.parameterStorePull(getHandle(), weights.length, keys, vals, priority);
    }

    @Override
    public KVStoreHandle getHandle() {
        return (KVStoreHandle) super.getHandle();
    }

    private static KVStoreHandle createdKVStore() {
        return JnaUtils.parameterStoreCreate("device");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        KVStoreHandle pointer = (KVStoreHandle) handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.parameterStoreClose(pointer);
        }
    }

    /** A helper to wrap the optimizer so it can be called by the MXNet KVStore. */
    private static final class OptimizerCallback extends MXKVStoreStrUpdater {

        private Optimizer optimizer;

        OptimizerCallback(Optimizer optimizer) {
            this.optimizer = optimizer;
        }

        /** {@inheritDoc} */
        @Override
        public void call(BytePointer key, NDArrayHandle recv, NDArrayHandle local, Pointer handle) {
            String parameterId = JnaUtils.getStringValue(key);
            // updater callback arguments order is: index, gradient, weight.
            try (MxNDManager manager = MxNDManager.getSystemManager().newSubManager()) {
                MxNDArray grad = manager.create(recv);
                MxNDArray weight = manager.create(local);
                grad.setShouldFree(false);
                weight.setShouldFree(false);
                optimizer.update(parameterId, weight, grad);
            }
        }
    }
}
