const express = require("express");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");
const fs = require("fs");
const cors = require("cors");
const Groq = require("groq-sdk"); // ✅ thay OpenAI bằng Groq
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// Bộ nhớ tạm tài liệu (chưa nhúng vector DB ở bản này)
let documentText = "";

// Upload file
app.post("/upload", upload.single("file"), async (req, res) => {
  const filePath = req.file.path;
  const ext = req.file.originalname.split(".").pop();

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

    documentText = text;
    res.json({ message: "Upload thành công", length: text.length });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Lỗi xử lý file" });
  } finally {
    fs.unlinkSync(filePath);
  }
});

// Hỏi đáp với Groq
app.post("/ask", async (req, res) => {
  const { question } = req.body;
  if (!documentText) {
    return res.status(400).json({ error: "Chưa upload tài liệu" });
  }

  try {
    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant", // model free & nhanh của Groq
      messages: [
        {
          role: "system",
          content: "Bạn là trợ lý, chỉ trả lời dựa trên tài liệu được cung cấp.",
        },
        {
          role: "user",
          content: `Tài liệu:\n${documentText}\n\nCâu hỏi: ${question}`,
        },
      ],
    });

    const answer = completion.choices[0].message.content;
    res.json({ answer });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Lỗi khi gọi Groq API" });
  }
});

app.listen(3001, () =>
  console.log("🚀 Backend Groq chạy tại http://localhost:3001")
);
