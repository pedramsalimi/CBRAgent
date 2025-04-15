import streamlit as st
import os
import json
import logging
from datetime import datetime
import uuid
import re
from typing import Annotated, List, Literal, TypedDict
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import AnyMessage, add_messages
from typing import Callable
from counterfactual import generate_counterfactual, predict_loan_approval


if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


instructions = """
### **Welcome to the Loan Application Assistant User Study**

#### **Scenario:**
You are **John Doe**, a **36-year-old** individual applying for a loan to purchase a new car. After submitting your application, you receive a notification that your loan has been **rejected**. You now seek assistance to improve your loan application and explore possible ways to improve it for future attempts.

#### **Your Task:**
- **Interact with the Assistant:** Engage in a conversation with our intelligent loan assistant to receive insights and suggestions regarding your loan application.
- **Explore Solutions:** If the initial suggestions don’t fully address your concerns or aren’t feasible for you, feel free to share more details or specify your preferences to receive alternative solutions.
- **Provide Feedback:** Your interactions will help us improve the assistant's capabilities and user experience.

#### **Hints for a Productive Interaction:**
- If a suggested change doesn’t align with your situation, you might consider specifying any limitations you have.
- Think about different aspects of your financial profile that you might want to adjust or maintain to enhance your loan approval chances.

#### **Interaction Guidelines:**
- You can interact with the assistant up to **5 times** to ensure a focused and effective experience.

#### **Post-Interaction:**
After your conversation with the assistant, you’ll be asked to complete a brief survey to share your feedback and experiences.
"""

MAX_INTERACTIONS = 5  

os.environ["OPENAI_API_KEY"] = "APIKEY"

if "global_chat" not in st.session_state:
    st.session_state.global_chat = []
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "agent_app" not in st.session_state:
    st.session_state.agent_app = None
if "user_constraints" not in st.session_state:
    st.session_state.user_constraints = []
if "loan_result" not in st.session_state:
    st.session_state.loan_result = None
if "counterfactual_result" not in st.session_state:
    st.session_state.counterfactual_result = None
if "informed_decision" not in st.session_state:
    st.session_state.informed_decision = False
if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0
if "survey_completed" not in st.session_state:
    st.session_state.survey_completed = False
if "participant_id" not in st.session_state:
    st.session_state.participant_id = None
if "participant_logger" not in st.session_state:
    st.session_state.participant_logger = None

feature_name_mapping = {
    "person_emp_length": "Employment Length (years)",
    "person_age": "Age",
    "cb_person_cred_hist_length": "Credit History Length (years)",
    "loan_amnt": "Loan Amount",
    "loan_percent_income": "Loan as Percentage of Income",
    "person_home_ownership": "Home Ownership Status",
    "person_income": "Income",
    "loan_intent": "Loan Intent",
    "loan_grade": "Loan Grade",
    "loan_int_rate": "Loan Interest Rate",
    "cb_person_default_on_file": "Default on File"
}

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_data: dict
    user_constraints: List[str]

def generate_participant_id():
    return str(uuid.uuid4())

def get_participant_logger(participant_id: str) -> logging.Logger:
    logger = logging.getLogger(participant_id)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_filename = os.path.join("logs", f"loan_application_chat_{participant_id}.log")
        fh = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def log_chat_message(role: str, message: str):
    """
    Log a chat message (user or assistant) in a structured JSON format.
    """
    if st.session_state.participant_logger:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "participant_id": st.session_state.participant_id,
            "role": role,
            "message": message
        }
        st.session_state.participant_logger.info(json.dumps(log_entry))

def save_final_conversation_and_survey():
    """
    Save the entire conversation and survey responses for further analysis.
    """
    participant_id = st.session_state.participant_id
    conversation_log = st.session_state.global_chat

    full_convo_filename = os.path.join("logs", f"full_conversation_{participant_id}.json")
    with open(full_convo_filename, "w", encoding="utf-8") as f:
        json.dump({
            "participant_id": participant_id,
            "conversation": conversation_log
        }, f, ensure_ascii=False, indent=2)


def collect_user_data():
    st.markdown(instructions)
    predefined_data = {
        "name": "John Doe",
        "person_age": 36,
        "person_income": 75000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "VENTURE",
        "loan_amnt": 20000,
        "loan_percent_income": 0.267,
        "cb_person_cred_hist_length": 10,
        "cb_person_default_on_file": "N",
        "loan_grade": "C",
        "loan_int_rate": 0.15
    }

    st.markdown("### **Your Loan Application Details**")
    st.markdown(f"""
    - **Name:** {predefined_data['name']}
    - **Age:** {predefined_data['person_age']}
    - **Income:** ${predefined_data['person_income']}
    - **Home Ownership Status:** {predefined_data['person_home_ownership']}
    - **Employment Length:** {predefined_data['person_emp_length']} years
    - **Loan Intent:** {predefined_data['loan_intent']}
    - **Desired Loan Amount:** ${predefined_data['loan_amnt']}
    - **Loan as Percentage of Income:** {predefined_data['loan_percent_income'] * 100}%
    - **Credit History Length:** {predefined_data['cb_person_cred_hist_length']} years
    - **Default on File:** {predefined_data['cb_person_default_on_file']}
    - **Loan Grade:** {predefined_data['loan_grade']}
    - **Loan Interest Rate:** {predefined_data['loan_int_rate'] * 100}%
    """)

    if st.button("Submit Application"):
        user_data = predefined_data.copy()

        st.session_state.user_data = user_data

        if not st.session_state.participant_id:
            st.session_state.participant_id = generate_participant_id()
            st.session_state.participant_logger = get_participant_logger(st.session_state.participant_id)
            st.session_state.participant_logger.info(f"Participant ID: {st.session_state.participant_id}")

        st.session_state.participant_logger.info(f"User Data Submitted: {user_data}")
        st.success("Application submitted!")

        assistant_message = f"Thank you {user_data['name']} for submitting your application. Let me process your information."
        st.session_state.global_chat.append({"assistant": assistant_message})
        log_chat_message("assistant", assistant_message)

        loan_result = predict_loan_approval(user_data)
        st.session_state.loan_result = loan_result
        st.session_state.participant_logger.info(f"Loan Prediction Result: {loan_result}")

        loan_status = 1.0 if loan_result == "Approved" else 0.0
        st.session_state.user_data['loan_status'] = loan_status

        st.session_state.informed_decision = True
        if loan_result == "Approved":
            assistant_response = f"Congratulations {user_data['name']}, your loan has been approved!"
        else:
            assistant_response = f"I'm sorry {user_data['name']}, your loan application was not approved. Is there anything I can assist you with?"
        st.session_state.global_chat.append({"assistant": assistant_response})
        log_chat_message("assistant", assistant_response)

def extract_constraints(user_input: str) -> list:
    constraints = []
    patterns = {
        "person_income": r"(?:cannot|can't|unable to|must not)\s+(?:change|modify|alter)\s+(?:my\s+)?income",
        "person_age": r"(?:cannot|can't|unable to|must not)\s+(?:change|modify|alter)\s+(?:my\s+)?age",
        "person_emp_length": r"(?:cannot|can't|unable to|must not)\s+(?:change|modify|alter)\s+(?:my\s+)?employment length",
        "loan_amnt": r"(?:cannot|can't|unable to|must not)\s+(?:change|modify|alter)\s+(?:my\s+)?loan amount"
    }

    for feature, pattern in patterns.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            constraints.append(feature)

    if constraints:
        st.session_state.participant_logger.info(f"Extracted Constraints from User Input: {constraints}")
    return constraints

import json

@tool
def generate_counterfactual_tool(input_str: str) -> str:
    """
    Use this tool to provide a counterfactual explanation as a solution (ways to improve 
    the user's loan application) for the current loan rejection.

    Parameters (passed within input_str as JSON):
    - user_data (dict): The user's original loan application data.
    - user_constraints (List[str]): List of features that are immutable.

    Returns:
    - str: A natural language explanation of the counterfactual or a message requesting 
           user constraints if needed.
    """
    data = json.loads(input_str)
    user_data = data["user_data"]
    user_constraints = data["user_constraints"]


    features = [
        'loan_status',  
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ]

    selected_ite_values = {
        'person_age': -0.0025391578674316,
        'person_income': 0.0123,
        'person_home_ownership': 0.240622043609619,
        'person_emp_length': 0.0952982902526855,
        'loan_intent': 0.680086612701416,
        'loan_grade': 0.304718017578125,
        'loan_amnt': -0.5441474914550781,
        'loan_int_rate': 2.384185791015625e-06,
        'loan_percent_income': 2.384185791015625e-06,
        'cb_person_default_on_file': 0.0001311302185058,
        'cb_person_cred_hist_length': 0.0054240226745605
    }

    ordered_ite_values = {feature: selected_ite_values.get(feature, 0.0) for feature in features}
    user_data["loan_status"] = 0
    ordered_user_data = {feature: user_data.get(feature, None) for feature in features}

    immutable_defaults = ["person_age", "cb_person_cred_hist_length"]
    for feature in immutable_defaults:
        if feature not in user_constraints:
            user_constraints.append(feature)

    counterfactual_result = generate_counterfactual(
        query_instance=ordered_user_data,
        ite_values=ordered_ite_values,
        constraints=user_constraints
    )

    if counterfactual_result is not None:
        suggested_changes = [k for k, v in counterfactual_result.items() if v != user_data.get(k)]
        immutable_features = [k for k in suggested_changes if k in user_constraints]

        if immutable_features:
            human_readable_features = [feature_name_mapping.get(feat, feat) for feat in immutable_features]
            return (
                f"I apologize for the inconvenience. The system suggests changes to: {', '.join(human_readable_features)}, "
                f"but you have indicated these features cannot be changed. "
                "Can we explore alternatives or confirm if any of these aspects can be modified?"
            )

        changed_features = {
            feature: value for feature, value in counterfactual_result.items()
            if feature not in user_constraints and user_data.get(feature) != value
        }

        if changed_features:
            explanations = []
            for feature, new_value in changed_features.items():
                original_value = user_data.get(feature)
                if new_value != original_value:
                    human_friendly = feature_name_mapping.get(feature, feature)
                    explanations.append(f"- {human_friendly}: Change from {original_value} to {new_value}.")

            if explanations:
                explanation_text = "To improve your loan approval chances, consider the following changes:\n" + "\n".join(explanations)
            else:
                explanation_text = "No changes are necessary based on the current data."

            return explanation_text
        else:
            return "No changes needed based on the current constraints."

    return "No counterfactual solution found based on the provided data and constraints."


@tool
def update_constraints_tool(input_str: str) -> str:
    """
    Extract and update the user's constraints based on their feedback, add them to the 
    list of constraints, and generate an updated counterfactual explanation.

    Parameters (passed within input_str as JSON):
    - immutable_features (List[str]): List of features the user cannot change.
    - user_data (dict): The user's original loan application data.

    Returns:
    - str: Updated counterfactual explanation or a message if no counterfactual is found.
    """
    data = json.loads(input_str)
    immutable_features = data["immutable_features"]
    user_data = data["user_data"]

    if 'user_constraints' not in st.session_state:
        st.session_state.user_constraints = []

    for feature in immutable_features:
        if feature not in st.session_state.user_constraints:
            st.session_state.user_constraints.append(feature)

    cf_input_str = json.dumps({
        "user_data": user_data,
        "user_constraints": st.session_state.user_constraints
    })
    counterfactual_explanation = generate_counterfactual_tool(cf_input_str)
    return counterfactual_explanation


class Assistant:
    def __init__(self, runnable: Callable):
        self.runnable = runnable

    def __call__(self, state: TypedDict, config: dict):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are an intelligent loan application assistant designed to help users understand the outcomes of their loan applications. Your primary functions include:
            - Generating actionable counterfactual explanations as solutions or ways for users to improve their loan applications for rejected loans using `generate_counterfactual_tool` in order to get the user's loan accepted.
            - Updating application constraints based on user feedback using `update_constraints_tool` and generating new solutions accordingly.

            You have access to the following tools:
            1. `generate_counterfactual_tool`: Use this tool when a loan application is rejected to provide users with actionable steps and improvements to increase the chances of loan acceptance.
            2. `update_constraints_tool`: Use this tool when users indicate that certain aspects of their application cannot be modified. It updates these immutable constraints and generates new solutions based on the updated constraints.

            You should invoke `generate_counterfactual_tool` in response to user queries seeking ways to improve or get their loan approved, such as:
            - "How can I improve my loan application?"
            - "How can I get my loan accepted?"
            - "Give me a solution for my loan rejection."
            - "What can I do to get my loan accepted?"
            - "Steps to enhance my loan approval chances."
            - "Advice on increasing my loan application's success."
            - "Suggestions to better my loan application."
            - "How to strengthen my loan request?"
            - "Ways to make my loan more likely to be approved."
            - "Improve my chances for loan approval."
            - "What steps can I take to secure my loan?"
            - "How can I increase my loan approval rate?"
            - "Best practices to get my loan approved."
            - "Methods to enhance my loan application's attractiveness."
            - "What improvements can I make to get my loan accepted?"
            - "Guidance on achieving loan approval."

            You should invoke `update_constraints_tool` when the user specifies that certain parts of their application are non-negotiable or cannot be changed. This tool will help adjust the constraints accordingly and provide new solutions based on these updated constraints.

            - Finally, if user seems happy with the provided solution be polite and ask if you can do any other thing for them.
            
            \n\nCurrent user data:\n<User Data>\n{user_data}\n</User Data>
            \nCurrent constraints:\n<Constraints>\n{user_constraints}\n</Constraints>
            \nCurrent time: {time}.''',
                    ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())



assistant_runnable = assistant_prompt | llm.bind_tools(
    [generate_counterfactual_tool, update_constraints_tool]
)

builder = StateGraph(State)
assistant = Assistant(assistant_runnable)
builder.add_node("assistant", assistant)

tools = [generate_counterfactual_tool, update_constraints_tool]
tool_node = ToolNode(tools)
builder.add_node("tools", tool_node)
builder.add_edge(START, "assistant")

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

builder.add_conditional_edges("assistant", should_continue)
builder.add_edge("tools", "assistant")

loan_graph = builder.compile(checkpointer=MemorySaver())

def display_chat():
    if st.session_state.user_data and st.session_state.loan_result and not st.session_state.survey_completed:
        st.subheader("Chat with the Assistant")
        messages_placeholder = st.empty()

        def render_messages():
            with messages_placeholder.container():
                for chat in st.session_state.global_chat:
                    if "user" in chat:
                        st.chat_message("user").write(chat["user"])
                    if "assistant" in chat:
                        st.chat_message("assistant").write(chat["assistant"])

        render_messages()

        prompt = st.chat_input("Ask a question or provide information")
        if prompt:
            st.session_state.interaction_count += 1
            st.session_state.global_chat.append({"user": prompt})
            log_chat_message("user", prompt)

            conversation_messages = []
            for chat in st.session_state.global_chat:
                if "user" in chat:
                    conversation_messages.append(HumanMessage(content=chat["user"]))
                if "assistant" in chat:
                    conversation_messages.append(AIMessage(content=chat["assistant"]))

            try:
                thread_id = str(uuid.uuid4())
                st.session_state.participant_logger.info(f"Invoking loan_graph with thread_id: {thread_id}")
                final_state = loan_graph.invoke(
                    {
                        "messages": conversation_messages,
                        "user_data": st.session_state.user_data,
                        "user_constraints": st.session_state.user_constraints
                    },
                    config={"configurable": {"thread_id": thread_id}}
                )
                agent_response = final_state["messages"][-1].content
                st.session_state.participant_logger.info(f"Agent Response: {agent_response}")

                for call in final_state.get("tool_calls", []):
                    st.session_state.participant_logger.info(f"Tool Call: {call.tool} with input: {call.input} returned: {call.output}")

                st.session_state.global_chat.append({"assistant": agent_response})
                log_chat_message("assistant", agent_response)

            except Exception as e:
                st.session_state.participant_logger.error(f"Error during graph invocation: {e}", exc_info=True)
                agent_response = "An error occurred while processing your request. Please try again later."
                st.session_state.global_chat.append({"assistant": agent_response})
                log_chat_message("assistant", agent_response)

            if st.session_state.interaction_count >= MAX_INTERACTIONS:
                st.session_state.survey_completed = True

            render_messages()

    if st.session_state.survey_completed:
        st.subheader("Survey")
        with st.form("survey_form"):
            st.markdown("### Please take a moment to complete the survey below:")

            satisfaction = st.slider(
                "1. Overall, how satisfied are you with the assistance provided by the chatbot?",
                1, 5, 3
            )
            clarity = st.slider(
                "2. How clear and understandable are the counterfactual explanations provided by the assistant?",
                1, 5, 3
            )
            helpfulness = st.slider(
                "3. How helpful are the suggestions in explaining what changes could improve your loan application?",
                1, 5, 3
            )
            fairness = st.slider(
                "4. How fair and ethically appropriate do you find the recommendations?",
                1, 5, 3
            )
            trustworthiness = st.slider(
                "5. How much do you trust the assistant’s advice?",
                1, 5, 3
            )


            feedback_positive = st.text_area("5. What did you like most about your interaction with the assistant?")
            feedback_negative = st.text_area("6. What did you dislike or find challenging?")
            suggestions = st.text_area("7. Suggestions for improvement.")

            submitted = st.form_submit_button("Submit Survey")
            if submitted:
                if not st.session_state.participant_id:
                    st.session_state.participant_id = generate_participant_id()
                    st.session_state.participant_logger = get_participant_logger(st.session_state.participant_id)
                    st.session_state.participant_logger.info(f"Participant ID: {st.session_state.participant_id}")

                survey_response = {
                    "participant_id": st.session_state.participant_id,
                    "timestamp": datetime.now().isoformat(),
                    "satisfaction": satisfaction,
                    "clarity": clarity,
                    "helpfulness": helpfulness,
                    "responsiveness": responsiveness,
                    "feedback_positive": feedback_positive,
                    "feedback_negative": feedback_negative,
                    "suggestions": suggestions
                }

                survey_filename = os.path.join("logs", f"survey_responses_{st.session_state.participant_id}.json")
                with open(survey_filename, "a", encoding="utf-8") as f:
                    f.write(json.dumps(survey_response) + "\n")
                st.session_state.participant_logger.info(f"Survey Response: {survey_response}")

                save_final_conversation_and_survey()

                st.success("Thank you for completing the survey!")

    elif st.session_state.user_data and st.session_state.loan_result and not st.session_state.survey_completed:
        st.info(f"Number of interactions: {st.session_state.interaction_count}/{MAX_INTERACTIONS}")
    elif not st.session_state.user_data:
        st.info("Please submit your loan application first.")

def main():
    st.title("Loan Application Assistant")
    if not st.session_state.user_data:
        collect_user_data()

    if st.session_state.agent_app is None and st.session_state.user_data:
        st.session_state.agent_app = loan_graph
        if st.session_state.participant_logger:
            st.session_state.participant_logger.info("Agent application initialized.")

    display_chat()

if __name__ == "__main__":
    main()
