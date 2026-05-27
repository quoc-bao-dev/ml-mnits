import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const docDir = path.join(process.cwd(), "document");

export async function GET() {
  try {
    if (!fs.existsSync(docDir)) {
      return NextResponse.json([]);
    }

    const files = fs.readdirSync(docDir)
      .filter((file) => file.endsWith(".md"))
      .sort(); // Sắp xếp theo tên file để đảm bảo thứ tự 01, 02, 03...

    const documents = files.map((file) => {
      const filePath = path.join(docDir, file);
      const content = fs.readFileSync(filePath, "utf-8");
      
      // Trích xuất tiêu đề từ dòng H1 đầu tiên (# Tiêu đề)
      const firstLine = content.split("\n")[0] || "";
      const title = firstLine.replace(/^#\s+/, "").trim() || file;

      return {
        key: file.replace(".md", ""),
        title,
        filename: file,
      };
    });

    return NextResponse.json(documents);
  } catch (error: any) {
    return NextResponse.json(
      { error: `Không thể đọc danh sách tài liệu: ${error.message}` },
      { status: 500 }
    );
  }
}
