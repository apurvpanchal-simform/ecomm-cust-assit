import chainlit as cl
@cl.on_message
async def main(msg):
    await cl.Message(author="sasass", content="Hello").send()
