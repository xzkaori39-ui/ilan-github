import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "i兰 · 校园知识服务助手",
  description: "面向校园多部门的可信制度知识服务助手",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
