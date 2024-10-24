import {openai} from "./openai.js";
import {Document} from "langchain/document";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
import {OpenAIEmbeddings} from "@langchain/openai";
import {CharacterTextSplitter} from "langchain/text_splitter";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {YoutubeLoader} from "langchain/document_loaders/web/youtube";

const question = process.argv[2] || "Hi";
console.log("question is", question);
const video = "https://youtu.be/zR_iuq2evXo?si=cG8rODgRgXOx9_Cn";

export const createStore = (docs) =>
    MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());

const docsFromYTVideo = (video) => {
    const loader = YoutubeLoader.createFromUrl(video, {
        language: "en",
        addVideoInfo: true,
    });

    return loader.loadAndSplit(
        new CharacterTextSplitter({
            separator: "",
            chunkSize: 2500,
            chunkOverlap: 100,
        })
    );
};

export const docsFromPDF = () => {
    const loader = new PDFLoader("xbox.pdf");
    return loader.loadAndSplit(
        new CharacterTextSplitter({
            separator: ". ",
            chunkSize: 2500,
            chunkOverlap: 200,
        })
    );
};

const loadStore = async () => {
    const videoDocs = await docsFromYTVideo(video);
    const pdfDocs = await docsFromPDF();

    return createStore([...videoDocs, ...pdfDocs]);
};

const query = async () => {
    const store = await loadStore();
    const results = await store.similaritySearch(question, 2);

    const response = await openai.chat.completions.create({
        model: "gpt-3.5-turbo-16k-0613",
        temperature: 0,
        messages: [
            {
                role: "assistant",
                content:
                    "You are a helpful AI assistant. Answser questions to your best ability.",
            },
            {
                role: "user",
                content: `Answer the following question using the provided context. If you cannot answer the question with the context, don't lie and make up stuff. Just say you need more context.
          Question: ${question}
    
          Context: ${results.map((r) => r.pageContent).join("\n")}`,
            },
        ],
    });
    console.log(
        `Answer: ${response.choices[0].message.content}\n\nSources: ${results
            .map((r) => r.metadata.source)
            .join(", ")}`
    );
};

query();
