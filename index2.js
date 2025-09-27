const express = require("express");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const mammoth = require("mammoth");
const fs = require("fs");
const cors = require("cors");
const Groq = require("groq-sdk"); // ✅ dùng Groq SDK
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// Bộ nhớ tạm (chưa dùng vector DB)
let documents = [];

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

    // Lưu tài liệu vào mảng
    documents.push({
      name: req.file.originalname,
      content: text,
      length: text.length,
    });

    res.json({
      message: "Upload thành công",
      totalDocs: documents.length,
      docs: documents.map((d) => ({ name: d.name, length: d.length })),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Lỗi xử lý file" });
  } finally {
    fs.unlinkSync(filePath); // Xóa file tạm
  }
});

/**
 * Xem danh sách tài liệu đã upload
 */
app.get("/documents", (req, res) => {
  res.json(
    documents.map((d, i) => ({
      id: i + 1,
      name: d.name,
      length: d.length,
    }))
  );
});

/**
 * Hỏi đáp với Groq dựa trên toàn bộ tài liệu
 */
app.post("/ask", async (req, res) => {
  const { question } = req.body;
  if (documents.length === 0) {
    return res.status(400).json({ error: "Chưa upload tài liệu nào" });
  }

  try {
    // Gộp tất cả tài liệu thành 1 string
    const allDocsText = documents
      .map((d) => `📄 ${d.name}:\n${d.content}`)
      .join("\n\n---\n\n");

    const completion = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant", // model free & nhanh của Groq
      messages: [
        {
          role: "system",
          content:
            "Bạn là trợ lý, chỉ trả lời dựa trên tài liệu được cung cấp. Nếu không thấy thông tin thì trả lời 'Không tìm thấy trong tài liệu'.",
        },
        {
          role: "user",
          content: `Các tài liệu:\n${allDocsText}\n\nCâu hỏi: ${question}`,
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
