from os import environ
from asyncio import run

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from dotenv import find_dotenv, load_dotenv

from model import make_prediction

load_dotenv(find_dotenv())


bot = Bot(environ.get('BOT_TOKEN'))
dp = Dispatcher()


async def start_handler(message: Message):
    await message.answer("Этот бот может предсказать колличество коментариев, которое будет под постом про транспорт в ВК. Просто скинь сюда текст поста:")

async def post_text_hanler(message: Message):
    await message.answer(f"Предсказанное количество коментариев: {make_prediction(message.text)}")

async def start(dp: Dispatcher):
    dp.message.register(start_handler, Command(commands='start'))
    dp.message.register(post_text_hanler)
    await dp.start_polling(bot)

if __name__ == "__main__":
    print('ready')
    run(start(dp))