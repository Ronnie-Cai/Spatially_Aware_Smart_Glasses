Spatially Aware Multimodal AI Assistant for Smart Glasses
Background and Objectives
Smart glasses provide a natural platform for an AI assistant that can see from the user’s point of view, listen to spoken questions, and respond through audio or a near-eye display. Recent multimodal large language models (MLLMs) can recognize objects, describe activities, and answer questions about images or video. However, recognizing the content of a single image is not the same as understanding a physical environment over time. A useful wearable assistant must also reason about spatial relationships, maintain a memory of previously observed places and objects, and respond with sufficiently low latency for natural interaction.

Conceptual smart-glasses assistant answering a spoken question about visible objects

Figure 1. Conceptual example of a user asking a smart-glasses assistant about objects in the current scene. This image was AI-generated.

This project aims to develop a voice-interactive AI assistant on smart glasses such as Rokid Glasses or RayNeo X3 Pro. The system will integrate the glasses’ camera, microphone, speaker, and display with an MLLM hosted on a local GPU server or accessed through a cloud API. The implementation may use either an end-to-end real-time multimodal model or a modular pipeline combining automatic speech recognition (ASR), visual understanding, language reasoning, and text-to-speech (TTS).

The project has two levels of objectives:

Core objective — real-time scene understanding: Build an end-to-end prototype that accepts a spoken question, observes the current first-person scene, and answers questions about visible objects, activities, and spatial relationships. Example questions include: “What objects are on the table?”, “What is the person in front of me doing?”, and “Which bottle is closer to me?”
Advanced objective — spatial reasoning with memory: Extend the assistant to reason over observations collected at different times and locations. Possible capabilities include answering “Where did I last see my keys?”, guiding the user back toward a previously observed object, or providing simple object-directed navigation in a controlled indoor environment.
Two-level project scope covering real-time scene understanding and spatial reasoning with memory

Figure 2. Overview of the core and advanced objectives of the proposed smart-glasses assistant. This image was AI-generated.

The second objective should be treated as an extension after a reliable perception-and-dialogue baseline is established. In particular, relative spatial judgments from the current camera view may be implemented using monocular cues or depth estimation, whereas persistent or metric spatial reasoning may require temporal tracking, camera pose or SLAM information, and an explicit spatial memory.

The final system should demonstrate not only a working wearable application, but also a systematic evaluation of its perception accuracy, spatial reasoning ability, response latency, robustness, and resource requirements. Because smart-glasses SDKs expose different capabilities, the chosen device and final scope will be confirmed through an early feasibility study of camera, audio, display, pose/SLAM, networking, and background-processing access.

Core Tasks
Study the SDK and development workflow of the selected smart glasses, and verify access to the camera, microphone, speaker/display, motion or pose data, and network communication.
Design the end-to-end system architecture linking the glasses to a local GPU server or multimodal model API.
Implement spoken interaction, including speech recognition, multimodal question answering, and speech or visual responses.
Develop a real-time scene-understanding baseline for object recognition, open-vocabulary scene description, and egocentric activity recognition.
Implement and evaluate spatial-relation queries such as left/right, in front of/behind, on/under, inside/near, and closer/farther from the user.
Construct a small, controlled evaluation set recorded from the glasses, containing representative objects, activities, viewpoints, occlusions, and spatial questions with ground-truth answers.
Evaluate task accuracy and end-to-end latency, and analyze common failure cases.
Advanced Tasks
Depending on progress, the student can select one of the following directions for further development and systematic evaluation:

Develop an agent layer that selects perception, retrieval, memory, and dialogue tools according to the user’s request.
Design an episodic memory that stores selected observations, timestamps, object identities, visual embeddings, and optional pose or location information without retaining unnecessary raw video.
Implement an advanced demonstration, such as last-seen object retrieval or object-directed guidance in a controlled indoor environment.
Compare the proposed memory or agent design with simple baselines, such as using only the latest frame, retaining a fixed window of frames, or retrieving past frames through visual-text similarity.
Develop uncertainty-aware responses that allow the assistant to express insufficient evidence, request another viewpoint, or avoid unreliable spatial guidance.
Add visual grounding so that language responses can be associated with the relevant object or region in the scene, with optional labels or highlights on the glasses display.
Explore object-directed navigation in a controlled environment. Any navigation function should be treated as a research prototype rather than a safety-critical aid.
Possible Research Questions
How accurately do current MLLMs understand spatial relationships in egocentric images or short videos, and which types of relations remain difficult?
Does combining an MLLM with explicit detectors, monocular depth estimation, tracking, or pose/SLAM information improve spatial answers compared with direct end-to-end prompting?
Which memory representation is most effective for a wearable assistant: stored keyframes, object-centric records, a temporal event graph, a 2D/3D semantic map, or a hybrid representation?
How should the assistant decide what to store, update, merge, or forget so that memory remains useful under limited storage, computation, and battery budgets?
Can structured retrieval using object, time, action, and location information outperform embedding-only retrieval for questions such as “Where did I last see this object?”
Can active perception improve reliability—for example, can the agent recognize insufficient evidence and ask the user to turn their head, move closer, or provide another view?
How should uncertainty be estimated and communicated so that the assistant avoids confidently giving incorrect spatial or navigation instructions?
How well does the system generalize across users, rooms, lighting conditions, object arrangements, and previously unseen object categories?
A student can select one primary research question for detailed experimentation. The other capabilities can support the prototype or serve as optional extensions.

Expected Background
Students must have basic programming experience in Python and an interest in multimodal AI or intelligent agents. Familiarity with LLM inference framework such as Huggingface Transformers, vllm, and calling LLM APIs will be helpful.

Experience in Android/Kotlin/Java or Unity/C# is desirable for glasses application development but is not required at the beginning. Depending on the selected research direction, students may also learn object detection and tracking, visual-language models, vector retrieval, scene graphs, monocular depth estimation, SLAM, and evaluation of interactive AI systems.

Students should be willing to work with physical hardware, diagnose SDK and networking issues, collect a small amount of consented egocentric data, and follow appropriate privacy and research-ethics practices.

Related Literature
EgoLife: Towards Egocentric Life Assistant provides recent motivation and data for multimodal assistants operating from wearable first-person video.
Reference guidelines and specification
The RayNeo Open Platform describes Android/Unity-oriented development and spatial capabilities for the X-series glasses.
The RayNeo X3 Pro development guide covers ADB setup, Android/Unity SDKs, native camera/audio access, and optional SLAM and plane-detection functions.
The Rokid Glasses specifications list an egocentric camera, microphone array, speakers, and near-eye display; third-party SDK access to the required streams should be verified before selecting this device.
Keywords
AI Glasses, Smart Glasses, Multimodal Large Language Models, Egocentric Vision, Spatial Perception, Spatial Reasoning, Scene Understanding, Activity Recognition, Embodied AI, AI Agents, Episodic Memory, Visual Retrieval, Human–AI Interaction
