#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace kokopop {

// Read-only memory mapping of a file. RAII; non-copyable; movable.
//
// On success: data() returns a pointer to size() bytes of file contents.
// On failure: ok() is false, error() contains a description; data() is null.
//
// Backed by mmap() on POSIX (macOS/Linux) and CreateFileMapping/MapViewOfFile
// on Windows. Falls back gracefully — callers should be prepared to use
// stream I/O when ok() is false (e.g. exotic filesystems, files > address space).
class FileMapping {
public:
    FileMapping() = default;
    explicit FileMapping(const std::string & path);

    FileMapping(const FileMapping &)             = delete;
    FileMapping & operator=(const FileMapping &) = delete;

    FileMapping(FileMapping && other) noexcept;
    FileMapping & operator=(FileMapping && other) noexcept;

    ~FileMapping();

    bool                ok()    const { return _data != nullptr; }
    const uint8_t *     data()  const { return _data; }
    size_t              size()  const { return _size; }
    const std::string & error() const { return _error; }

    // Hint the kernel about expected access patterns. No-op on Windows.
    // call_after_init=true switches from SEQUENTIAL hint to RANDOM after the
    // initial linear read, so subsequent on-demand pages don't pre-fetch
    // pointlessly.
    void advise_random();

private:
    void close();

    const uint8_t * _data = nullptr;
    size_t          _size = 0;
    std::string     _error;

#if defined(_WIN32)
    void * _file_handle    = nullptr;  // HANDLE
    void * _mapping_handle = nullptr;  // HANDLE
#else
    int _fd = -1;
#endif
};

} // namespace kokopop
