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

import ai.djl.mxnet.javacpp.SymbolHandle;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.mxnet.jna.NativeResource;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;
import ai.djl.util.Utils;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * {@code Symbol} is an internal helper for symbolic model graphs used by the {@link
 * ai.djl.nn.SymbolBlock}.
 *
 * @see ai.djl.nn.SymbolBlock
 * @see <a href="https://mxnet.incubator.apache.org/api/python/docs/api/symbol/index.html">MXNet
 *     Symbol</a>
 */
public class Symbol extends NativeResource {

    private String[] outputs;
    private MxNDManager manager;

    /**
     * Constructs a {@code Symbol}.
     *
     * @param manager the manager to attach the symbol to
     * @param pointer the symbol's native data location
     */
    Symbol(MxNDManager manager, SymbolHandle pointer) {
        super(pointer);
        this.manager = manager;
        manager.attach(getUid(), this);
    }

    /**
     * Loads a symbol from a path.
     *
     * @param manager the manager to load the symbol to
     * @param path the path to the symbol file
     * @return the new symbol
     */
    public static Symbol load(MxNDManager manager, String path) {
        SymbolHandle pointer = JnaUtils.createSymbolFromFile(path);
        return new Symbol(manager, pointer);
    }

    /**
     * Returns the symbol argument names.
     *
     * @return the symbol argument names
     */
    public String[] getArgNames() {
        return JnaUtils.listSymbolArguments(getHandle());
    }

    /**
     * Returns the MXNet auxiliary states for the symbol.
     *
     * @return the MXNet auxiliary states for the symbol
     */
    public String[] getAuxNames() {
        return JnaUtils.listSymbolAuxiliaryStates(getHandle());
    }

    /**
     * Returns the symbol names.
     *
     * @return the symbol names
     */
    public String[] getAllNames() {
        return JnaUtils.listSymbolNames(getHandle());
    }

    /**
     * Returns the symbol outputs.
     *
     * @return the symbol outputs
     */
    public String[] getOutputNames() {
        if (outputs == null) {
            outputs = JnaUtils.listSymbolOutputs(getHandle());
        }
        return outputs;
    }

    private String[] getInternalOutputNames() {
        return JnaUtils.listSymbolOutputs(getInternals().getHandle());
    }

    /**
     * Copies the symbol.
     *
     * @return a new copy of the symbol
     */
    public Symbol copy() {
        throw new UnsupportedOperationException("Not implemented yet");
    }

    /**
     * Returns the output symbol by index.
     *
     * @param index the index of the output
     * @return the symbol output as a new symbol
     */
    public Symbol get(int index) {
        SymbolHandle pointer = JnaUtils.getSymbolOutput(getInternals().getHandle(), index);
        return new Symbol(manager, pointer);
    }

    /**
     * Returns the output symbol with the given name.
     *
     * @param name the name of the symbol to return
     * @return the output symbol
     * @throws IllegalArgumentException Thrown if no output matches the name
     */
    public Symbol get(String name) {
        String[] out = getInternalOutputNames();
        int index = Utils.indexOf(out, name);
        if (index < 0) {
            throw new IllegalArgumentException("Cannot find output that matches name: " + name);
        }
        return get(index);
    }

    /**
     * Returns the symbol internals.
     *
     * @return the symbol internals symbol
     */
    public Symbol getInternals() {
        SymbolHandle pointer = JnaUtils.getSymbolInternals(getHandle());
        return new Symbol(manager, pointer);
    }

    /**
     * Returns the list of names for all internal outputs.
     *
     * @return a list of names
     */
    public List<String> getLayerNames() {
        String[] outputNames = getInternalOutputNames();
        String[] allNames = getAllNames();
        Set<String> allNamesSet = new LinkedHashSet<>(Arrays.asList(allNames));
        // Kill all params field and keep the output layer
        return Arrays.stream(outputNames)
                .filter(n -> !allNamesSet.contains(n))
                .collect(Collectors.toList());
    }

    /**
     * Infers the shapes for all parameters inside a symbol from the given input shapes.
     *
     * @param pairs the given input name and shape
     * @return a map of arguments with names and shapes
     */
    public Map<String, Shape> inferShape(PairList<String, Shape> pairs) {
        List<List<Shape>> shapes = JnaUtils.inferShape(this, pairs);
        if (shapes == null) {
            throw new IllegalArgumentException("Cannot infer shape based on the data provided!");
        }
        List<Shape> argShapes = shapes.get(0);
        List<Shape> outputShapes = shapes.get(1);
        List<Shape> auxShapes = shapes.get(2);
        // TODO: add output to the map
        String[] argNames = getArgNames();
        String[] auxNames = getAuxNames();
        String[] outputNames = getOutputNames();
        Map<String, Shape> shapesMap = new ConcurrentHashMap<>();
        for (int i = 0; i < argNames.length; i++) {
            shapesMap.put(argNames[i], argShapes.get(i));
        }
        for (int i = 0; i < auxNames.length; i++) {
            shapesMap.put(auxNames[i], auxShapes.get(i));
        }
        for (int i = 0; i < outputNames.length; i++) {
            shapesMap.put(outputNames[i], outputShapes.get(i));
        }
        return shapesMap;
    }

    @Override
    public SymbolHandle getHandle() {
        return (SymbolHandle) super.getHandle();
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return Arrays.toString(getOutputNames());
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        SymbolHandle pointer = (SymbolHandle) handle.getAndSet(null);
        if (pointer != null) {
            manager.detach(getUid());
            JnaUtils.freeSymbol(pointer);
            manager = null;
        }
    }
}
