from langchain_ollama import ChatOllama
from autogen import ConversableAgent, GroupChat, GroupChatManager


## add config for ollama server
base_url = "http://localhost:11434/v1"
model = 'llama3.2:1b'

llm_config = {
    'base_url': base_url,
    'api_key': "fake-key",
    "temperature": 0.4,
    'model': model
}

# Different Agent for different task in software planning
# Architect agent
architect_agent = ConversableAgent(
    name="Architect_Agent",
    system_message="You provide the best decision options for the software design.",
    llm_config=llm_config,
    description="Provides architecture and design options.",
)

# Backend developer agent

backend_agent = ConversableAgent(
    name="Backend_Agent",
    system_message="You suggest the best backend API/microservices code design for the given software design.",
    llm_config=llm_config,
    description="Suggests backend API/microservices options.",
)

# Front end developer agent
uI_agent = ConversableAgent(
    name="UI_Agent",
    system_message="You recommend best user interface and user experience design and code for proposed software design.",
    llm_config=llm_config,
    description="Recommends User interface and user experience.",
)

# Cloud operation agent
cloud_agent = ConversableAgent(
    name="Cloud_Agent",
    system_message="You suggest the best cloud infrastructure and suggest deployment of proposed software.",
    llm_config=llm_config,
    description="Recommends cloud Deployment design, requirements, and CI/CD code .",
)

# Cost agent to find cost of software project
cost_agent = ConversableAgent(
    name="Cost_Agent",
    system_message="You provide the cost for resources like architect, development, cloud, testing effort and deployment.",
    llm_config=llm_config,
    description="Provides cost of proposed software from end-to-end.",
)


# Create a Group Chat
group_chat_with_introductions = GroupChat(
    agents=[architect_agent, backend_agent, uI_agent, cloud_agent, cost_agent],
    messages=[],
    max_round=6,
    send_introductions=True,  # Send system messages to introduce each agent
)

# Create a Group Chat Manager
group_chat_manager_with_intros = GroupChatManager(
    groupchat=group_chat_with_introductions, llm_config=llm_config
)

# Create a software planner agent
software_planner_agent = ConversableAgent(
    name="Software_Planner_Agent",
    system_message="You summarize the all the software plan provided by the group chat.",
    llm_config=llm_config,
    description="Summarizes the software plan.",
)

# Initiate a chat with the group chat manager
chat_result = software_planner_agent.initiate_chats(
    [
        {
            "recipient": group_chat_manager_with_intros,
            "message": "I'm planning a software that take input as current stock price of tesla and predict the stock price after 5 years. Can you help me design, code, user interface, cloud deployment and cost of the software? I will start working from today itself.",
            "summary_method": "reflection_with_llm",
        },
        {
            "recipient": group_chat_manager_with_intros,
            "message": "Please refine the software design plan with additional details.",
            "summary_method": "reflection_with_llm",
        },
    ]
)

# Finding cost of agent, token and other details 
for result in chat_result:    
    print(result.cost)