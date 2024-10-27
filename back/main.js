import express from "express";
import {openai} from "./openai.js";
import cors from "cors";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
import {OpenAIEmbeddings} from "@langchain/openai";
import {CharacterTextSplitter} from "@langchain/textsplitters";
import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf";
import {YoutubeLoader} from "@langchain/community/document_loaders/web/youtube";

const app = express();
app.use(express.json());
app.use(cors({origin: "*"}));

const port = 3000;

// بارگذاری و تقسیم اسناد ویدئو YouTube
const docsFromYTVideo = async (video) => {
    try {
        const loader = YoutubeLoader.createFromUrl(video, {
            language: "en",
            addVideoInfo: true,
        });
        return await loader.loadAndSplit(
            new CharacterTextSplitter({
                separator: " ",
                chunkSize: 2500,
                chunkOverlap: 100,
            })
        );
    } catch (error) {
        console.error("YouTube transcription not found:", error.message);
        return [];
    }
};

// بارگذاری و تقسیم اسناد PDF
const docsFromPDF = () => {
    const loader = new PDFLoader("xbox.pdf");
    return loader.loadAndSplit(
        new CharacterTextSplitter({
            separator: ". ",
            chunkSize: 2500,
            chunkOverlap: 200,
        })
    );
};

// بارگذاری store سند
const loadStore = async () => {
    const videoDocs = await docsFromYTVideo("https://www.youtube.com/watch?v=fmIKhp_Mnb8");
    const pdfDocs = await docsFromPDF();
    return MemoryVectorStore.fromDocuments([...videoDocs, ...pdfDocs], new OpenAIEmbeddings());
};

// ایجاد store برای پرسش و پاسخ‌ها
const storePromise = loadStore();

// دریافت پاسخ با استفاده از اسناد
const queryWithDocs = async (store, question, history) => {
    const results = await store.similaritySearch(question, 2);
    const context = results.map((r) => r.pageContent).join("\n");

    const response = await openai.chat.completions.create({
        model: "gpt-4",
        temperature: 0,
        messages: [
            ...history,
            {
                role: "system",
                content:
                    "You are a helpful AI assistant. Answer questions using the provided context if available.",
            },
            {
                role: "user",
                content: `Answer the following question using the provided context. If you cannot answer with the context, ask for more context.
                    Question: ${question}
                    Context: ${context}`,
            },
        ],
    });

    return response.choices[0].message.content;
};

// API endpoint برای دریافت پیام و پاسخ با استفاده از سند
app.post("/message", async (req, res) => {
    const {message} = req.body;
    const store = await storePromise;
    const history = [
        {
            role: "system",
            content: "Hi, you're an AI assistant. Answer questions to the best of your ability based on available context.",
        },
    ];

    const responseContent = await queryWithDocs(store, message, history);
    res.json({ai: responseContent});
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
