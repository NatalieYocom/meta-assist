import subprocess

def run(command_text: str, speak) -> str:
    speak("Committing...")

    status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
    if not status.stdout.strip():
        return "Nothing to commit. Working tree is clean."

    subprocess.run(["git", "add", "."], check=True)
    res = subprocess.run(["git", "commit", "-m", "Auto-commit via Meta"], capture_output=True, text=True)

    if res.returncode == 0:
        return "Done."
    return "Commit failed. Check the console for details."
