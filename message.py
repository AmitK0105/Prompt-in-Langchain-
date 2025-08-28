from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1", task="text-generation", max_new_tokens=100)

model= ChatHuggingFace(llm=llm)

message= [ SystemMessage(content= "You are a helpful assistant"),
          HumanMessage(content= "Tell me about langchain")]

result= model.invoke(message)
message.append(AIMessage(content= result.content))

print(message)

