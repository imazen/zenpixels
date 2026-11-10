//! Shared thread-scoped allocation-failure injection for integration tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;

// Const initialization has no allocation or destructor, so allocator entry
// can safely read the calling thread's scope. Harness threads remain unaffected.
thread_local! {
    static THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
}

fn threshold() -> usize {
    THRESHOLD.try_with(Cell::get).unwrap_or(usize::MAX)
}

/// Retains fixture serialization; failure injection itself is thread-local.
static FAIL_LOCK: Mutex<()> = Mutex::new(());

struct FailOver;

// SAFETY: `alloc`/`alloc_zeroed`/`realloc` return null past the threshold
// (callers observe failure, the contract's only requirement); smaller
// allocations and all `dealloc`s delegate to `System` with the same layout.
unsafe impl GlobalAlloc for FailOver {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > threshold() {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > threshold() {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > threshold() {
            return core::ptr::null_mut();
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static FAIL_OVER: FailOver = FailOver;

/// Restore the threshold on drop, even when a test assertion panics while it
/// is lowered.
struct ThresholdGuard(usize);

impl Drop for ThresholdGuard {
    fn drop(&mut self) {
        THRESHOLD.with(|threshold| threshold.set(self.0));
    }
}

/// Fail every allocation above `bytes` while `body` runs, then restore.
pub(super) fn failing_above<T>(bytes: usize, body: impl FnOnce() -> T) -> T {
    let previous = THRESHOLD.with(|threshold| threshold.replace(bytes));
    let _guard = ThresholdGuard(previous);
    body()
}

pub(super) fn lock() -> std::sync::MutexGuard<'static, ()> {
    FAIL_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
