#include "core/file_mapping.h"

#include <utility>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  include <cerrno>
#  include <cstring>
#endif

namespace kokopop {

FileMapping::FileMapping(const std::string & path) {
#if defined(_WIN32)
    HANDLE file = CreateFileA(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        _error = "FileMapping: CreateFileA failed for " + path;
        return;
    }
    LARGE_INTEGER fsize{};
    if (!GetFileSizeEx(file, &fsize) || fsize.QuadPart <= 0) {
        _error = "FileMapping: GetFileSizeEx failed for " + path;
        CloseHandle(file);
        return;
    }
    HANDLE mapping = CreateFileMappingA(
        file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        _error = "FileMapping: CreateFileMappingA failed for " + path;
        CloseHandle(file);
        return;
    }
    void * addr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (addr == nullptr) {
        _error = "FileMapping: MapViewOfFile failed for " + path;
        CloseHandle(mapping);
        CloseHandle(file);
        return;
    }
    _file_handle    = file;
    _mapping_handle = mapping;
    _data           = static_cast<const uint8_t *>(addr);
    _size           = static_cast<size_t>(fsize.QuadPart);
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        _error = "FileMapping: open() failed for " + path + ": " + std::strerror(errno);
        return;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        _error = "FileMapping: fstat() failed for " + path + ": " + std::strerror(errno);
        ::close(fd);
        return;
    }
    const size_t size = static_cast<size_t>(st.st_size);
    void * addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        _error = "FileMapping: mmap() failed for " + path + ": " + std::strerror(errno);
        ::close(fd);
        return;
    }

    // We read the whole file once, linearly, during load — request sequential
    // prefetch behavior. advise_random() may be called afterwards.
    ::madvise(addr, size, MADV_SEQUENTIAL);
    _fd   = fd;
    _data = static_cast<const uint8_t *>(addr);
    _size = size;
#endif
}

FileMapping::FileMapping(FileMapping && other) noexcept {
    *this = std::move(other);
}

FileMapping & FileMapping::operator=(FileMapping && other) noexcept {
    if (this == &other) return *this;
    close();
    _data           = other._data;
    _size           = other._size;
    _error          = std::move(other._error);
#if defined(_WIN32)
    _file_handle    = other._file_handle;
    _mapping_handle = other._mapping_handle;
    other._file_handle    = nullptr;
    other._mapping_handle = nullptr;
#else
    _fd             = other._fd;
    other._fd       = -1;
#endif
    other._data = nullptr;
    other._size = 0;
    return *this;
}

FileMapping::~FileMapping() {
    close();
}

void FileMapping::close() {
#if defined(_WIN32)
    if (_data) {
        UnmapViewOfFile(_data);
        _data = nullptr;
    }
    if (_mapping_handle) {
        CloseHandle(_mapping_handle);
        _mapping_handle = nullptr;
    }
    if (_file_handle) {
        CloseHandle(_file_handle);
        _file_handle = nullptr;
    }
#else
    if (_data && _size) {
        ::munmap(const_cast<uint8_t *>(_data), _size);
    }
    _data = nullptr;
    if (_fd >= 0) {
        ::close(_fd);
        _fd = -1;
    }
#endif
    _size = 0;
}

void FileMapping::advise_random() {
#if !defined(_WIN32)
    if (_data && _size) {
        ::madvise(const_cast<uint8_t *>(_data), _size, MADV_RANDOM);
    }
#endif
}

} // namespace kokopop
