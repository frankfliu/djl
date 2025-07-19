import org.gradle.api.Project
import org.gradle.api.file.Directory
import org.gradle.internal.impldep.org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.gradle.internal.impldep.org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.URI
import java.net.URL
import java.nio.file.Path
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.zip.GZIPInputStream
import java.util.zip.ZipInputStream

operator fun File.div(other: String) = File(this, other)
operator fun Directory.div(other: String): File = file(other).asFile

val nowFormatted: String
    get() = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"))

infix fun URL.into(file: File) {
    file.outputStream().use { out ->
        openStream().use { `in` -> `in`.copyTo(out) }
    }
}

infix fun URL.gzipInto(file: File) {
    file.outputStream().use { out ->
        GZIPInputStream(openStream()).use { `in` -> `in`.copyTo(out) }
    }
}

infix fun URL.zipInto(dir: File) {
    ZipInputStream(openStream()).use { zis ->
        var entry = zis.nextEntry
        while (entry != null) {
            validateArchiveEntry(entry.name, dir.toPath())
            val entryFile = File(dir, entry.name)
            if (entry.isDirectory) {
                entryFile.mkdirs()
            } else {
                entryFile.parentFile?.mkdirs()
                FileOutputStream(entryFile).use { fos ->
                    zis.copyTo(fos)
                }
            }
            entry = zis.nextEntry
        }
    }
}

infix fun URL.tarInto(dir: File) {
    println("=========")
    TarArchiveInputStream(GzipCompressorInputStream(openStream())).use { tis ->
        var entry = tis.nextEntry
        while (entry != null) {
            validateArchiveEntry(entry.name, dir.toPath())
            val entryFile = File(dir, entry.name)
            if (entry.isDirectory) {
                entryFile.mkdirs()
            } else {
                entryFile.parentFile?.mkdirs()
                FileOutputStream(entryFile).use { fos ->
                    tis.copyTo(fos)
                }
            }
            entry = tis.nextEntry
        }
    }
}

fun validateArchiveEntry(name: String, destination: Path) {
    if (name.contains("..")) {
        throw IOException("Invalid archive entry")
    }
    val path = destination.resolve(name).toAbsolutePath().normalize()
    if (!path.startsWith(destination.normalize())) {
        throw IOException("Invalid archive entry")
    }
}

var File.text
    get() = readText()
    set(value) = writeText(value)

val URL.text
    get() = readText()

val osName = System.getProperty("os.name")
val os = osName.lowercase()
val arch = System.getProperty("os.arch")
val home = System.getProperty("user.home")

val String.url: URL
    get() = URI(this).toURL()


// provide directly a Directory instead of a DirectoryProperty
val Project.buildDirectory: Directory
    get() = layout.buildDirectory.get()
