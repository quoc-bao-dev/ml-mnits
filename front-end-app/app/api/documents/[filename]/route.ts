import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const docDir = path.join(process.cwd(), "document");

export async function GET(
  request: Request,
  { params }: { params: Promise<{ filename: string }> }
) {
  try {
    const { filename } = await params;
    const cleanFilename = filename.endsWith(".md") ? filename : `${filename}.md`;
    const filePath = path.join(docDir, cleanFilename);

    // Bảo mật: Ngăn chặn Directory Traversal bằng cách kiểm tra đường dẫn an toàn
    const resolvedPath = path.resolve(filePath);
    const resolvedDocDir = path.resolve(docDir);
    if (!resolvedPath.startsWith(resolvedDocDir)) {
      return NextResponse.json({ error: "Yêu cầu không hợp lệ" }, { status: 403 });
    }

    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ error: "Không tìm thấy tài liệu" }, { status: 404 });
    }

    const content = fs.readFileSync(filePath, "utf-8");
    return NextResponse.json({ content });
  } catch (error: any) {
    return NextResponse.json(
      { error: `Lỗi đọc tài liệu: ${error.message}` },
      { status: 500 }
    );
  }
}
