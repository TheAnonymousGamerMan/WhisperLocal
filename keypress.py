"""Read a single keypress without needing Enter -- cross-platform."""


def read_key(prompt=""):
    if prompt:
        print(prompt, end="", flush=True)
    try:
        import msvcrt
        key = msvcrt.getch()
        try:
            key = key.decode("utf-8", errors="ignore")
        except Exception:
            key = ""
        print()
        return key
    except ImportError:
        pass

    try:
        import sys
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()
        return key
    except Exception:
        return input().strip()[:1] or " "
