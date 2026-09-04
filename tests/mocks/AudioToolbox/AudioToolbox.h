#pragma once
#include <cstdint>
using UInt32 = uint32_t;
using OSStatus = int32_t;
constexpr OSStatus noErr = 0;
constexpr UInt32 kAudioFormatLinearPCM = 1;
constexpr UInt32 kAudioFormatFlagIsSignedInteger = 2;
constexpr UInt32 kAudioFormatFlagIsPacked = 4;
constexpr UInt32 kAudioQueueProperty_IsRunning = 5;
struct AudioStreamBasicDescription {
    double mSampleRate;
    UInt32 mFormatID, mFormatFlags, mBytesPerPacket, mFramesPerPacket,
           mBytesPerFrame, mChannelsPerFrame, mBitsPerChannel;
};
struct AudioQueueBuffer { void * mAudioData; UInt32 mAudioDataByteSize; };
using AudioQueueBufferRef = AudioQueueBuffer *;
struct MockAudioQueue;
using AudioQueueRef = MockAudioQueue *;
struct AudioStreamPacketDescription {};
using AudioQueueOutputCallback = void (*)(void *, AudioQueueRef, AudioQueueBufferRef);
OSStatus AudioQueueNewOutput(const AudioStreamBasicDescription *, AudioQueueOutputCallback,
                            void *, void *, void *, UInt32, AudioQueueRef *);
OSStatus AudioQueueAllocateBuffer(AudioQueueRef, UInt32, AudioQueueBufferRef *);
OSStatus AudioQueueEnqueueBuffer(AudioQueueRef, AudioQueueBufferRef, UInt32, const void *);
OSStatus AudioQueueStart(AudioQueueRef, const void *);
OSStatus AudioQueueStop(AudioQueueRef, bool);
OSStatus AudioQueueDispose(AudioQueueRef, bool);
OSStatus AudioQueueGetProperty(AudioQueueRef, UInt32, void *, UInt32 *);
