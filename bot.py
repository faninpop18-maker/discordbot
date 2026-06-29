import discord
import asyncio
import datetime
import sys
import os
import shutil
import aiohttp
from PIL import Image
import numpy as np
from io import BytesIO

TOKEN = "TOKEN"
CHANNEL_ID = ID 
LOG_DIR = "botslogs"
LOG_FILE = os.path.join(LOG_DIR, "logs.txt")
MAX_LOG_SIZE = 10 * 1024 * 1024
MAX_LOG_FILES = 10
ASCII_WIDTH = 60

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma2:2b"

def image_to_colored_ascii(image_bytes, width=ASCII_WIDTH):
    try:
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")
        w_percent = width / float(img.size[0])
        h_size = int((float(img.size[1]) * float(w_percent)) / 1.8)
        img = img.resize((width, h_size), Image.Resampling.LANCZOS)
        pixels = np.array(img)
        chars = [" ", "░", "▒", "▓", "█"]
        result = ""
        for row in pixels:
            for r, g, b in row:
                brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
                char_index = int(brightness / 255 * (len(chars) - 1))
                char = chars[char_index]
                result += f"\033[48;2;{r};{g};{b}m{char}\033[0m"
            result += "\n"
        return result
    except Exception as e:
        return f"[Ошибка конвертации: {e}]"

async def ask_ollama(prompt: str) -> str:
    # Ограничение длины запроса
    if len(prompt) > 1000000:
        return "Слишком длинный запрос. Сократи до 100000 символов."

    # Системный промпт (русский язык)
    system_prompt = "Ты — полезный ассистент. Отвечай только на русском языке. Всегда используй русский язык в своих ответах."
    full_prompt = f"{system_prompt}\n\nВопрос: {prompt}\n\nОтвет:"

    async with aiohttp.ClientSession() as session:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": 150,
                "temperature": 0.7
            }
        }
        try:
            async with session.post(OLLAMA_URL, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "Пустой ответ от модели.")
                else:
                    return f"Ошибка API: {response.status}"
        except asyncio.TimeoutError:
            return "Модель слишком долго думает. Попробуй позже."
        except aiohttp.ClientError as e:
            return f"Ошибка подключения к Ollama: {e}"

class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_time = datetime.time(hour=12, minute=0)
        self.channel_id = CHANNEL_ID
        self.daily_task = None
        self.log_file = None
        self.user_cooldown = {}

    def rotate_logs(self):
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > MAX_LOG_SIZE:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            archive_name = os.path.join(LOG_DIR, f"logs_{timestamp}.txt")
            shutil.move(LOG_FILE, archive_name)
            print(f"Лог заархивирован: {archive_name}")
            archives = [f for f in os.listdir(LOG_DIR) if f.startswith("logs_") and f.endswith(".txt")]
            if len(archives) > MAX_LOG_FILES:
                archives.sort()
                for old_file in archives[:-MAX_LOG_FILES]:
                    os.remove(os.path.join(LOG_DIR, old_file))
                    print(f"Удалён старый архив: {old_file}")

    async def on_ready(self):
        print(f"Бот {self.user} запущен.")
        self.rotate_logs()
        self.log_file = open(LOG_FILE, "a", encoding="utf-8")
        print(f"Лог сохраняется в: {LOG_FILE}")
        print("Вводите сообщения в терминале для отправки в канал.")
        print("Для выхода введите 'exit'.")
        print(f"ИИ-модель: {OLLAMA_MODEL} (команда !ask)")
        if self.daily_task is None:
            self.daily_task = asyncio.create_task(self.send_daily_message())
        asyncio.create_task(self.read_terminal())

    async def on_message(self, message):
        if message.author == self.user:
            return

        # Команда !ask (с анти-спамом)
        if message.content.startswith("!ask "):
            if message.author.id in self.user_cooldown:
                time_left = (self.user_cooldown[message.author.id] - datetime.datetime.now()).total_seconds()
                if time_left > 0:
                    await message.reply(f"Подожди {int(time_left)+1} сек, не спамь!")
                    return
            self.user_cooldown[message.author.id] = datetime.datetime.now() + datetime.timedelta(seconds=10)

            prompt = message.content[5:].strip()
            if not prompt:
                await message.reply("Напиши вопрос после !ask")
                return
            async with message.channel.typing():
                response = await ask_ollama(prompt)
                if len(response) > 1900:
                    response = response[:1900] + "..."
                await message.reply(response)
            return

        # Логирование сообщений (только из нужного канала)
        is_target_channel = message.channel.id == self.channel_id
        is_thread_of_target = (
            message.channel.type in (discord.ChannelType.public_thread, discord.ChannelType.private_thread)
            and message.channel.parent_id == self.channel_id
        )

        if is_target_channel or is_thread_of_target:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{timestamp}] {message.author.display_name}: {message.content}"

            if message.attachments:
                attachment_lines = []
                for att in message.attachments:
                    if att.content_type and att.content_type.startswith("image/"):
                        image_data = await att.read()
                        colored_ascii = image_to_colored_ascii(image_data, width=ASCII_WIDTH)
                        attachment_lines.append(f"IMAGE: {att.filename}\n{colored_ascii}")
                    else:
                        attachment_lines.append(f"FILE: {att.filename} ({att.size//1024} КБ)")
                log_line += "\n" + "\n".join(attachment_lines)

            print(f"\n{log_line}", flush=True)
            if self.log_file:
                self.log_file.write(log_line + "\n")
                self.log_file.flush()

    async def send_daily_message(self):
        await self.wait_until_ready()
        while not self.is_closed():
            now = datetime.datetime.now()
            next_run = datetime.datetime.combine(now.date(), self.target_time)
            if now.time() >= self.target_time:
                next_run += datetime.timedelta(days=1)
            wait_seconds = (next_run - now).total_seconds()
            print(f"Следующая отправка через {wait_seconds:.0f} секунд", flush=True)
            await asyncio.sleep(wait_seconds)
            channel = self.get_channel(self.channel_id)
            if channel is not None:
                await channel.send("сохранено")
                print("Сообщение 'сохранено' отправлено.", flush=True)
            else:
                print(f"Канал с ID {self.channel_id} не найден.", flush=True)

    async def read_terminal(self):
        loop = asyncio.get_event_loop()
        while not self.is_closed():
            line = await loop.run_in_executor(None, lambda: sys.stdin.readline().strip())
            if line is None:
                break
            if line.lower() == "exit":
                print("Завершение по команде exit", flush=True)
                await self.close()
                break
            if line:
                channel = self.get_channel(self.channel_id)
                if channel is not None:
                    await channel.send(line)
                    print(f"Отправлено: {line}", flush=True)
                else:
                    print(f"Канал с ID {self.channel_id} не найден.", flush=True)

    async def close(self):
        if self.log_file:
            self.log_file.close()
        await super().close()

intents = discord.Intents.default()
intents.message_content = True
client = MyClient(intents=intents)
client.run(TOKEN)
