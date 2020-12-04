/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.http;

import ai.djl.ModelException;
import ai.djl.modality.Input;
import ai.djl.serving.execution.PredictionService;
import ai.djl.serving.util.NettyUtils;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class handling inbound HTTP requests for the management API. */
public class InterferencePredictionRequestHandler extends HttpRequestHandler {

    private static final Logger logger =
            LoggerFactory.getLogger(InterferencePredictionRequestHandler.class);

    private static final Pattern PATTERN = Pattern.compile("^/predictions([/?].*)?");

    private PredictionService predictionService;
    private RequestParser requestParser;

    public InterferencePredictionRequestHandler(PredictionService predictionService) {
        this.predictionService = predictionService;
        this.requestParser = new RequestParser();
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            return PATTERN.matcher(req.uri()).matches();
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        if (segments.length < 3) {
            throw new ResourceNotFoundException();
        }
        if (HttpMethod.OPTIONS.equals(req.method())) {
            NettyUtils.sendJsonResponse(ctx, "{}");
        }
        Input input = requestParser.parseRequest(ctx, req, decoder);

        predictionService
                .predict(input, segments[2])
                .exceptionally(
                        t -> {
                            // TODO refactor: this can return ErrorResponseObject and then Accepts
                            // can handle this error-object as a normal result. to clarify: how to
                            // set the status
                            NettyUtils.sendError(ctx, t);
                            return null;
                        })
                .thenAccept(result -> NettyUtils.sendFullResponse(ctx, result));
    }
}
