// index.js
const express = require("express");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");
const fs = require("fs");
const cors = require("cors");
const Groq = require("groq-sdk");
require("dotenv").config();

let pipeline;
let embedder;

// Load embedding model bằng dynamic import (chỉ 1 lần)
(async () => {
  try {
    ({ pipeline } = await import("@xenova/transformers"));
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    console.log("✅ Embedding model loaded");
  } catch (err) {
    console.error("❌ Lỗi load model:", err);
  }
})();

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// Bộ nhớ tài liệu (chung cho tất cả user)
let documents = []; // [{id, name, chunks: [{text, embedding}]}]

let chatbotConfig = {
  model: "llama-3.1-8b-instant",
  temperature: 0.2,
  maxTokens: 1024,
};

/**
 * 📌 Chia văn bản thành chunks nhỏ
 */
function chunkText(text, size = 500) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

/**
 * 📌 Tính cosine similarity
 */
function cosineSim(vecA, vecB) {
  let dot = 0.0,
    normA = 0.0,
    normB = 0.0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * ============================
 * 📂 API cho Admin
 * ============================
 */

/**
 * Upload file
 */
app.post("/upload", upload.single("file"), async (req, res) => {
  const filePath = req.file.path;
  const ext = req.file.originalname.split(".").pop().toLowerCase();

  try {
    let text = "";
    if (ext === "pdf") {
      const dataBuffer = fs.readFileSync(filePath);
      const data = await pdfParse(dataBuffer);
      text = data.text;
    } else if (ext === "docx") {
      const data = await mammoth.extractRawText({ path: filePath });
      text = data.value;
    } else if (ext === "txt") {
      text = fs.readFileSync(filePath, "utf8");
    } else {
      return res.status(400).json({ error: "Định dạng file không hỗ trợ" });
    }

    // Chia nhỏ + embedding
    const chunks = chunkText(text);
    const chunkEmbeddings = [];
    for (const chunk of chunks) {
      const output = await embedder(chunk, {
        pooling: "mean",
        normalize: true,
      });
      const vec = Array.from(output.data);
      chunkEmbeddings.push({ text: chunk, embedding: vec });
    }

    documents.push({
      id: documents.length + 1,
      name: req.file.originalname,
      chunks: chunkEmbeddings,
    });

    res.json({
      message: "Upload thành công",
      totalDocs: documents.length,
      docs: documents.map((d) => ({
        id: d.id,
        name: d.name,
        chunks: d.chunks.length,
      })),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Lỗi xử lý file" });
  } finally {
    fs.unlinkSync(filePath);
  }
});

/**
 * Danh sách tài liệu
 */
app.get("/documents", (req, res) => {
  res.json(
    documents.map((d) => ({
      id: d.id,
      name: d.name,
      chunks: d.chunks.length,
    }))
  );
});

/**
 * Xóa tài liệu
 */
app.delete("/documents/:id", (req, res) => {
  const id = parseInt(req.params.id);
  const before = documents.length;
  documents = documents.filter((d) => d.id !== id);

  if (documents.length === before) {
    return res.status(404).json({ error: "Không tìm thấy tài liệu để xóa" });
  }
  res.json({ message: "Đã xóa tài liệu", totalDocs: documents.length });
});

/**
 * Reset toàn bộ
 */
app.post("/documents/sync", (req, res) => {
  documents = [];
  res.json({ message: "Đã reset dữ liệu", totalDocs: 0 });
});

/**
 * Config chatbot
 */
app.get("/config", (req, res) => {
  res.json(chatbotConfig);
});

app.post("/config", (req, res) => {
  const { model, temperature, maxTokens } = req.body;
  if (model) chatbotConfig.model = model;
  if (temperature !== undefined) chatbotConfig.temperature = temperature;
  if (maxTokens !== undefined) chatbotConfig.maxTokens = maxTokens;

  res.json({ message: "Đã cập nhật cấu hình", config: chatbotConfig });
});

/**
 * ============================
 * 👥 API cho User
 * ============================
 */

/**
 * Tra cứu tài liệu theo tên
 */
app.get("/search", (req, res) => {
  const q = (req.query.q || "").toString().toLowerCase();
  if (!q) {
    return res.status(400).json({ error: "Thiếu từ khóa tìm kiếm" });
  }

  const results = documents
    .filter((d) => d.name.toLowerCase().includes(q))
    .map((d) => ({
      id: d.id,
      name: d.name,
      chunks: d.chunks.length,
    }));

  if (results.length === 0) {
    return res.status(404).json({ error: "Không tìm thấy tài liệu nào" });
  }

  res.json({ results });
});

/**
 * Người dùng hỏi đáp
 */
app.post("/ask", async (req, res) => {
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: "Thiếu câu hỏi" });

  if (documents.length === 0) {
    return res
      .status(400)
      .json({ error: "Chưa có tài liệu nào trong hệ thống" });
  }

  try {
    // Embedding câu hỏi
    const qVecOut = await embedder(question, {
      pooling: "mean",
      normalize: true,
    });
    const queryVec = Array.from(qVecOut.data);

    // Tìm top 5 đoạn liên quan
    let scored = [];
    for (const doc of documents) {
      for (const ch of doc.chunks) {
        const score = cosineSim(queryVec, ch.embedding);
        scored.push({ text: ch.text, score, doc: doc.name });
      }
    }
    scored.sort((a, b) => b.score - a.score);
    const topChunks = scored.slice(0, 5);

    const context = topChunks
      .map((c, i) => `(${i + 1}) [${c.doc}]: ${c.text}`)
      .join("\n\n");

    // Gọi Groq
    const completion = await groq.chat.completions.create({
      model: chatbotConfig.model,
      temperature: chatbotConfig.temperature,
      max_tokens: chatbotConfig.maxTokens,
      messages: [
        {
          role: "system",
          content: "Bạn là trợ lý, chỉ trả lời dựa trên tài liệu được cung cấp.",
        },
        {
          role: "user",
          content: `Ngữ cảnh:\n${context}\n\nCâu hỏi: ${question}`,
        },
      ],
    });

    const answer = completion.choices[0].message.content;
    res.json({ answer, usedChunks: topChunks });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Lỗi khi gọi Groq API" });
  }
});

/**
 * ============================
 * 🚀 Chạy server
 * ============================
 */
app.listen(3001, () =>
  console.log("🚀 Backend Groq RAG chạy tại http://localhost:3001")
);
