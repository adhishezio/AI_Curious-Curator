# The Curious Curator: An Interactive AI Encyclopedia

The Curious Curator is a Python-based project that creates an intelligent agent capable of visually identifying real-world objects, learning about them from user-provided sources, and engaging in detailed, conversational Q&A about them. It serves as a powerful, local, and personal encyclopedia for your immediate physical environment.

This project is currently **in an active development stage**. The core pipeline is functional, but further enhancements to its learning capabilities, performance, and physical embodiment are planned.

---

## Core Technologies

This project stands on three technological pillars to create a complete multimodal experience:

* **Vision & Recognition:** Utilizes the **CLIP** model from OpenAI to generate powerful embeddings (numerical fingerprints) for images, allowing the system to understand and compare what it sees. The system identifies objects by finding the closest visual match in its memory.

* **Knowledge & Memory:** Implements a **Retrieval-Augmented Generation (RAG)** pipeline.
    * **ChromaDB** is used as a local vector database to store the "memories" of the objects.
    * Knowledge is **user-curated**, allowing for hyper-specific information to be added for each object, moving beyond generic web searches.

* **Language & Reasoning:** Powered by a quantized **Google Gemma** model running locally via the `llama-cpp-python` library. This allows for fast, efficient, and private conversational AI without relying on cloud APIs.

---

## How It Works

The project is broken down into a simple, three-step workflow:

### 1. Capture & Teach
Run the `capture_image.py` script. This activates your webcam and allows you to take pictures of new objects you want the AI to learn about. When you save an image, you give the object a name (e.g., "logitech_mouse"), which creates a category for it.

### 2. Curate & Learn
Run the `build_knowledge_base.py` script. For each new object category, the system will prompt you to provide a source of knowledge, such as a specific Wikipedia page or a product's technical specification URL. It fetches this information, processes it, and stores it in the vector database, linking the detailed text to the object's visual appearance.

### 3. Converse & Query
Run the main `query_knowledge_base.py` script. This activates the full pipeline:
* It identifies an object shown to the webcam.
* It announces what it sees and prompts you for a question.
* You can ask a specific question in the terminal (e.g., "What is the battery life?").
* The system retrieves the curated knowledge for that object and uses the LLM to generate a precise, context-aware answer.

---

## Setup and Installation

To get this project running on your local machine, follow these steps.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/adhishezio/AI_Curious-Curator.git
    cd AI_Curious-Curator
    ```

2.  **Create a Python Virtual Environment**
    It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Install all the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Important Note for Windows Users (C++ Compiler)**
    The `llama-cpp-python` library requires a C++ compiler to be installed. If the `pip install` command fails, you will need to install the **Microsoft C++ Build Tools**. You can do this by downloading the "Visual Studio Installer", and selecting the "Desktop development with C++" workload.

---

## Future Development

This project serves as a strong foundation for several advanced features planned for future development:

* **DJI Tello Drone Integration:** The ultimate goal is to mount this "brain" onto a mobile platform, allowing the agent to autonomously explore and learn about its environment.
* **True Few-Shot Learning:** Implementing a dedicated Few-Shot Object Detection (FSOD) model to enable more robust and dynamic on-the-fly learning of new objects without needing to re-build the knowledge base.
* **Advanced Data Ingestion:** Expanding the knowledge curation module to handle more complex documents like PDFs (e.g., product manuals) and other unstructured data sources.
* **Graphical User Interface (GUI):** Creating a user-friendly interface to replace the current command-line interaction.
