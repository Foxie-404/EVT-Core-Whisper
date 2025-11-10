import os
import signal
import argparse
import random
import uvicorn
from typing import Union, Optional, Any
from fastapi import FastAPI
from PyEasyUtils import isPortAvailable, findAvailablePorts, terminateProcess

from transcribe import Voice_Transcribing


parser = argparse.ArgumentParser()
parser.add_argument("--host", help = "主机地址", type = str, default = "localhost")
parser.add_argument("--port", help = "端口",     type = int, default = 8080)
args = parser.parse_known_args()[0]

host = args.host
port = args.port if isPortAvailable(args.port, host) else random.choice(findAvailablePorts((8000, 8080)))


app = FastAPI()


@app.post("/terminate")
async def terminate():
    terminateProcess(os.getpid())


@app.get("/asr")
async def asr(
    modelPath: str = './Models/.pt',
    audioDir: str = './WAV_Files',
    verbose: Any = True,
    language: str = None,
    addLanguageInfo: bool = True,
    conditionOnPreviousText: Any = False,
    fp16: Any = True,
    outputRoot: str = './',
    outputDirName: str = 'SRT_Files'
):
    asr = Voice_Transcribing(
        modelPath, audioDir, verbose, language, addLanguageInfo, conditionOnPreviousText, fp16, outputRoot, outputDirName
    )
    return asr.transcribe()


if __name__ == "__main__":
    uvicorn.run(app, host = host, port = port)