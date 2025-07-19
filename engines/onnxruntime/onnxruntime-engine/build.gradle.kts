import kotlin.io.path.moveTo

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.onnxruntime"

dependencies {
    api(project(":api"))
    api(libs.onnxruntime)

    testImplementation(project(":testing"))
    testImplementation(project(":engines:pytorch:pytorch-engine"))
    testImplementation(project(":extensions:tokenizers"))
    testImplementation("com.microsoft.onnxruntime:onnxruntime-extensions:${libs.versions.onnxruntimeExtensions.get()}")

    testRuntimeOnly(libs.slf4j.simple)
}

sourceSets.main {
    java {
        srcDirs("src/main/java", "build/generated-src")
    }
}

tasks {
    val basePath = "${project.projectDir}/build/resources/main/nlp"
    val logger = project.logger
    processResources {
        outputs.dir(basePath)
        doLast {
            val url = "https://mlrepo.djl.ai/model/nlp"
            val tasks = listOf(
                "fill_mask",
                "question_answer",
                "text_classification",
                "text_embedding",
                "token_classification",
                "zero_shot_classification"
            )
            for (task in tasks) {
                val file = File("$basePath/$task/ai.djl.huggingface.onnxruntime.json")
                if (file.exists())
                    logger.lifecycle("model zoo metadata already exists: $task")
                else {
                    logger.lifecycle("Downloading model zoo metadata: $task")
                    file.parentFile.mkdirs()
                    "$url/$task/ai/djl/huggingface/onnxruntime/models.json.gz".url gzipInto file
                }
            }
        }
    }

    val copyGenAIsrc by registering {
        outputs.dir(buildDirectory / "generated-src")
        outputs.cacheIf { true }
        doLast {
            val ortGenaiVersion = libs.versions.onnxruntimeGenai.get()
            val tmp = buildDirectory / "download"
            tmp.mkdirs()
            val gitubUrl = "https://github.com/microsoft/onnxruntime-genai/releases/download/v$ortGenaiVersion"
            val files = listOf(
                "${gitubUrl}/onnxruntime-genai-${ortGenaiVersion}-linux-x64-cuda.tar.gz",
                "${gitubUrl}/onnxruntime-genai-${ortGenaiVersion}-linux-x64.tar.gz",
                "${gitubUrl}/onnxruntime-genai-${ortGenaiVersion}-osx-arm64.tar.gz",
                "${gitubUrl}/onnxruntime-genai-${ortGenaiVersion}-win-x64-zip",
                "https://github.com/microsoft/onnxruntime-genai/archive/refs/tags/v${ortGenaiVersion}.zip"
            )
            for (f in files) {
                println("Downloading: $f")
                if (f.endsWith(".zip")) {
                    f.url.zipInto(tmp)
                } else {
                    f.url.tarInto(tmp)
                }
            }
            val target = buildDirectory / "generated-src"
            target.deleteRecursively()
            (tmp / "onnxruntime-genai-${ortGenaiVersion}/src/java/src/main/java").toPath().moveTo(target.toPath())
        }
    }

    compileJava {
        options.apply {
            release = 8
            encoding = "UTF-8"
            compilerArgs = listOf("-proc:none", "-Xlint:all,-options,-static,-serial", "-Werror")
        }
        dependsOn(copyGenAIsrc)
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            artifactId = "onnxruntime-engine"
            pom {
                name = "DJL Engine Adapter for ONNX Runtime"
                description = "Deep Java Library (DJL) Engine Adapter for ONNX Runtime"
                url = "http://www.djl.ai/engines/onnxruntime/${project.name}"
            }
        }
    }
}
