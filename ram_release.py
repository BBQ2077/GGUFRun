"""模型載入完成後，把 llama-server 已經複製進顯卡的模型檔頁面交還 Windows。

mmap 載入時，放上 GPU 的權重先從模型檔讀進 RAM 再複製到顯存；Windows 版 llama.cpp
之後不會放掉這些頁面，它們一直留在行程的 working set 裡，和顯存裡那份重複。
EmptyWorkingSet 把整個 working set 交還系統：模型檔頁面是唯讀映射，變成可隨時回收的
檔案快取；CPU 層之後用到的頁面會從快取（或 SSD）重新映射回來，GPU 那份不會再被讀到。
"""
import ctypes
from ctypes import wintypes

_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010
_PROCESS_SET_QUOTA = 0x0100

# 這些模式的權重是行程私有記憶體或已鎖定，修剪只會把它們擠進分頁檔
_SKIP_FLAGS = {'--no-mmap', '-nmp', '--mlock'}
_SKIP_LOAD_MODES = {'mlock', 'mmap+mlock', 'none', 'dio'}


class _Counters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def eligible(command):
    """指令走 mmap（llama.cpp 預設 auto 或指定 mmap）才回傳 True。"""
    for index, token in enumerate(command):
        flag, equal, value = token.partition('=')
        if flag in _SKIP_FLAGS:
            return False
        if flag in ('-lm', '--load-mode'):
            if not equal:
                value = command[index + 1] if index + 1 < len(command) else ''
            if value.strip().lower() in _SKIP_LOAD_MODES:
                return False
    return True


def _working_set(kernel32, handle):
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    if kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return counters.WorkingSetSize
    return None


def release(pid):
    """回傳 (成功, 釋放前 GiB, 釋放後 GiB, 錯誤說明)。"""
    if not hasattr(ctypes, 'windll'):
        return False, None, None, '只支援 Windows'
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.K32EmptyWorkingSet.argtypes = (wintypes.HANDLE,)
    kernel32.K32GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(_PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ | _PROCESS_SET_QUOTA, False, pid)
    if not handle:
        return False, None, None, f'OpenProcess 失敗（錯誤碼 {ctypes.get_last_error()}）'
    try:
        before = _working_set(kernel32, handle)
        if not kernel32.K32EmptyWorkingSet(handle):
            return False, None, None, f'EmptyWorkingSet 失敗（錯誤碼 {ctypes.get_last_error()}）'
        after = _working_set(kernel32, handle)
        gib = lambda b: None if b is None else b / 2 ** 30
        return True, gib(before), gib(after), ''
    finally:
        kernel32.CloseHandle(handle)
