import asyncio
import json
import os
import nest_asyncio
from dotenv import load_dotenv
import sys 
from typing import List
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Función de anidado para uso en Jupyter
nest_asyncio.apply()
load_dotenv()

class MCP_ChatBot:
    def __init__(self):
        self.session: ClientSession = None
        
        # Configuración de modelo de lenguaje
        base_url = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("API_KEY", "dummy")
        self.model_name = os.getenv("MODEL_NAME", "qwen/qwen3.5-35b-a3b")
        
        print(f"🔗 Conectando a LLM en: {base_url}")
        print(f"🧠 Usando modelo: {self.model_name}\n")

        # Cliente asíncrono de OpenAI
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        self.available_tools: List[dict] = []
        self.messages: List[dict] = []

    async def process_query(self, query: str):
        # 1. Append the new user query to the persistent history
        self.messages.append({"role": "user", "content": query})
        process_query = True

        while process_query:
            print("🤔 Pensando...", end="\r")
            
            # Historial de chat 
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=self.available_tools,
                tool_choice="auto",
                max_tokens=2048,
                temperature=0.0
            )
            print(" " * 30, end="\r") # Clear "Thinking..."

            choice = response.choices[0]
            message = choice.message

            if message.tool_calls:
                # Historial de uso de herramientas
                formatted_tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    } for tc in message.tool_calls
                ]
                
                # Almacenar llamado de herramientas en historial de mensajes 
                self.messages.append({
                    "role": "assistant", 
                    "content": message.content,
                    "tool_calls": formatted_tool_calls
                })

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    tool_id = tool_call.id

                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            print(f"\n❌ Fallo en cargar argumentos de herramienta: {raw_args}")
                            tool_args = {}
                    else:
                        tool_args = raw_args

                    print(f"\n🛠️ Llamando herramienta: {tool_name} with args: {tool_args}")
                    
                    result = await self.session.call_tool(tool_name, arguments=tool_args)
                    
                    result_text = "\n".join([
                        c.text for c in result.content if hasattr(c, 'text')
                    ])

                    print(f"✅ Output de herramienta: {result_text[:150]}{'...' if len(result_text) > 150 else ''}")

                    # 4. Guardar output de herramienta en historial
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result_text
                    })
            else:
                # 5. Si no se llama herramientas, imprimir respuesta de modelo 
                if message.content:
                    print(f"\n🤖 Assistant: {message.content}")
                    self.messages.append({
                        "role": "assistant",
                        "content": message.content
                    })
                
                process_query = False

    async def chat_loop(self):
        print("\n🚀 Bienvenido al MCP Chatbot")
        print("Escribe tus preguntas o bien 'quit' para salir.\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ['quit', 'exit']:
                    break
                if not query:
                    continue
                    
                await self.process_query(query)
                print("\n" + "-"*50 + "\n")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    async def connect_to_server_and_run(self):
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "research_mcp.py"], # Ensure this matches your server filename
            env=os.environ.copy(),
        )
        
        print("Conectando a servidor MCP...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
    
                response = await session.list_tools()
                tools = response.tools
                print(f"✅ Conectado a MCP. Herramientas disponibles: {[tool.name for tool in tools]}\n")
                
                # Conversión de esquema MCP -> OpenAI
                self.available_tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema 
                    }
                } for tool in tools]
    
                await self.chat_loop()

async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())
