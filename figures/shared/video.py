"""Composite playback and cursors onto the very same rendered PDF panel."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw


def render_video(panel: dict, png: Path, geometry: list[dict], output: Path,
                 *, fps=25, scale=2.0) -> None:
    base = Image.open(png).convert('RGB')
    # libx264 needs even dimensions; pad outside the PDF's own coordinates.
    size = tuple(n + n % 2 for n in base.size)
    background = Image.new('RGB', size, 'white')
    background.paste(base, (0, 0))
    start, end = panel['window']
    duration = end - start
    origin = float(panel['case'].get('media_time_origin_s', 0.0))
    count = math.ceil(duration * fps)
    media = next((item for item in geometry
                  if item['kind'] == 'media' and item['modality'] == 'video'), None)
    decoder = None
    if media:
        x, y, width, height = [round(media[key] * scale) for key in ('x', 'y', 'width', 'height')]
        decoder = subprocess.Popen([
            'ffmpeg', '-v', 'error', '-ss', repr(origin + start),
            '-i', panel['media_sources']['video'], '-t', repr(duration),
            '-vf', f'fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:white',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-']
    audio = panel.get('media_sources', {}).get('audio')
    if audio:
        command += ['-ss', repr(origin + start), '-i', audio, '-map', '0:v', '-map', '1:a:0', '-c:a', 'aac']
    command += ['-t', repr(duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(count):
            frame = background.copy()
            if decoder:
                nbytes = width * height * 3
                pixels = bytearray()
                while len(pixels) < nbytes:
                    chunk = decoder.stdout.read(nbytes - len(pixels))
                    if not chunk:
                        raise ValueError('source media ends before the requested playback window')
                    pixels.extend(chunk)
                moving = Image.frombytes('RGB', (width, height), bytes(pixels))
                frame.paste(moving, (x, y))
            draw = ImageDraw.Draw(frame)
            # Encoding frame k occurs at start + k/fps on the panel's seconds axis.
            fraction = (index / fps) / duration
            for row in geometry:
                if row['kind'] in ('track', 'media'):
                    cursor = (row['x'] + fraction * row['width']) * scale
                    top, bottom = row['y'] * scale, (row['y'] + row['height']) * scale
                    draw.line((cursor, top, cursor, bottom), fill='#d84f4f', width=2)
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f'ffmpeg failed encoding {output}')
        if decoder and decoder.wait() != 0:
            raise RuntimeError('ffmpeg failed decoding selected media')
    except BaseException:
        encoder.kill()
        encoder.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        if decoder:
            if decoder.poll() is None:
                decoder.kill()
            decoder.wait()
            decoder.stdout.close()


def still_video(png: Path, output: Path, duration=5.0) -> None:
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-i', str(png),
                    '-t', str(duration), '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(output)], check=True)
