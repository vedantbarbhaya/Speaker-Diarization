import argparse
import io
import os
from pathlib import Path
import select
import shutil
import subprocess as sp
import sys
from typing import Optional, IO


class AudioSeparator:
    def __init__(self, file_path: str, out_path: str = None, model: str = "htdemucs",
                 two_stems: Optional[str] = "vocals"):
        self.file_path = Path(file_path)
        self.out_path = Path(out_path) if out_path else self.file_path.parent / "temp-outputs"
        self.model = model
        self.two_stems = two_stems
        self.validate_directories()

    def validate_directories(self):
        """Ensure the output directory exists."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file {self.file_path} does not exist.")
        if not self.file_path.is_file():
            raise Exception(f"The path {self.file_path} is not a file.")
        self.out_path.mkdir(parents=True, exist_ok=True)

    def separate(self):
        """Runs Demucs to separate the audio file according to the specified model and options."""
        cmd = [
            "python3", "-m", "demucs.separate", "-n", self.model,
            "--two-stems", self.two_stems, str(self.file_path), "-o", str(self.out_path)
        ]

        print("Executing command: ", " ".join(cmd))
        process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        self.copy_process_streams(process)
        process.wait()
        if process.returncode != 0:
            print(f"Command failed for file {self.file_path}, something went wrong.")

    def copy_process_streams(self, process: sp.Popen):
        """Copies the process streams to standard output and error."""

        def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
            assert stream is not None
            if isinstance(stream, io.BufferedIOBase):
                stream = stream.raw
            return stream

        p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)

        stream_by_fd = {
            p_stdout.fileno(): (p_stdout, sys.stdout),
            p_stderr.fileno(): (p_stderr, sys.stderr),
        }
        print(stream_by_fd)
        fds = list(stream_by_fd.keys())

        while fds:
            ready = select.select(fds, [], [])[0]
            for fd in ready:
                p_stream, std = stream_by_fd[fd]
                raw_buf = p_stream.read(2 ** 16)
                if not raw_buf:
                    fds.remove(fd)
                    continue
                buf = raw_buf.decode()
                std.write(buf)
                std.flush()


# Example usage with command-line arguments
if __name__ == "__main__":
    file_path = '/content/audio-files/Donald Trump_ The Art of the Insult-[AudioTrimmer.com].mp3'
    out_path = '/content/audio-files/temp-outputs'
    separator = AudioSeparator(file_path=file_path, out_path=out_path)
    separator.separate()
