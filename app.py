import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.llms import openai
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from tabula.io import read_pdf
import fitz
from IPython.display import display
from IPython.display import Markdown
from PIL import Image


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    Try finding answer to every question possible.
    The basic questions would be like "What is the PDF about?", "What does the PDF tells about?" and 
    many more basic questions.
    If authors are in the pdf, questions could be asked related to authors.
    Try capturing title for every doc uploaded.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.5)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# def handle_user_query(query):
#     response = st.session_state.conversation({'question': query})
#     answer_paragraph = response["answer"].split("Helpful Answer:")[1].strip()
#     st.write(answer_paragraph)



# def handle_user_query(query):
#     response = st.session_state.conversation({'question': query})
#     answer_paragraph = response["answer"].split("Helpful Answer:")[1].strip()
#     if "Follow Up Input:" in answer_paragraph:
#         # Split the answer paragraph based on "Follow Up Input:"
#         parts = answer_paragraph.split("Follow Up Input:")
#         follow_up_input = parts[1].split("Standalone question:")
#         formatted_answer = "\n\nFollow Up Input:\n" + follow_up_input[0].strip() + "\n\nStandalone question:\n" + follow_up_input[1].strip()
#         formatted_answer = formatted_answer.replace("Answer:", "\n\nAnswer:")
#         st.write(formatted_answer)
#     else:
#         st.write("Answer: ", answer_paragraph)


def fetch_tables_and_images(pdf_path):
    #tables
    dfs  = read_pdf(pdf_path, pages='all', multiple_tables=True)
    folder_path = os.path.dirname(pdf_path)
    table_folder = os.path.join(folder_path, "tables")
    os.makedirs(table_folder, exist_ok=True)
    for i, df in enumerate(dfs):
        df.to_csv(os.path.join(table_folder, f"table_no_{i}.csv"))


    #images
    image_folder = os.path.join(folder_path, "images")
    os.makedirs(image_folder, exist_ok=True)

    pdf_document = fitz.open(pdf_path)
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_name = f"image_page_{page_number + 1}index{image_index}.png"
            image_path = os.path.join(image_folder, image_name)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
    st.success("Data have been stored in the folder.")


def fetch_and_display_image_info(folder_path):
    parent_directory = os.path.dirname(folder_path)
    # st.write(parent_directory)
    image_folder = os.path.join(parent_directory, "images")
    # st.write(image_folder)
    st.header("Image Information")
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        st.subheader(f"Image: {image_file}")
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # image_data = input_image_setup(image_path)
        # response = get_gemini_response(input_prompt, image_data, input_prompt)
        model = genai.GenerativeModel("gemini-1.5-flash")
        # def to_markdown(text):
        #     text = text.replace("•", "  *")
        #     return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))
        # response = get_gemini_response(input, image, input_prompt)
        response = model.generate_content(["Describe in detail the image", image], stream=True)
        response.resolve()
        # response = to_markdown(response.text)
        st.subheader("Generated Context:")
        st.write(response.text)




def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write(response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    st.header("Ask a Question from the PDF Files")

    user_question = st.text_input("Question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("PDF To Data Converter")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


        folder_path = st.text_input("Enter the Path for the PDF")
        if st.button("Fetch Data?"):
            with st.spinner("Fetching..."):
                fetch_tables_and_images(folder_path)
                # image_folder = os.path.join(folder_path, "images")


    st.header("Describe Images?")          
    if st.button("Generate"):
            with st.spinner("Generating..."):
                fetch_and_display_image_info(folder_path)


if __name__ == "__main__":
    main()