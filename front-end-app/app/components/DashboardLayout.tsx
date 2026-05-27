"use client";

import React, { useState, useEffect } from "react";
import { Layout, Menu, ConfigProvider, theme as antTheme } from "antd";
import {
  FileTextOutlined,
  ExperimentOutlined,
  RocketOutlined,
  ThunderboltOutlined,
  BookOutlined,
  DeploymentUnitOutlined,
} from "@ant-design/icons";
import type { MenuProps } from "antd";
import { usePathname, useRouter } from "next/navigation";

const { Sider, Header, Content } = Layout;

interface DocInfo {
  key: string;
  title: string;
  filename: string;
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [documents, setDocuments] = useState<DocInfo[]>([]);

  // Tải danh sách tài liệu để sinh menu sidebar
  useEffect(() => {
    fetch("/api/documents")
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setDocuments(data);
        }
      })
      .catch((err) => console.error("Error loading documents index:", err));
  }, []);

  // Xác định selectedKey dựa trên URL pathname
  const getSelectedKey = (): string => {
    if (pathname === "/ml/training") return "ml-training";
    if (pathname === "/ml/predict") return "ml-predict";
    if (pathname.startsWith("/doc/")) {
      const slug = pathname.replace("/doc/", "");
      return `doc-${slug}`;
    }
    return "ml-training"; // Mặc định
  };

  const selectedKey = getSelectedKey();

  // Định nghĩa Menu sidebar
  const menuItems: MenuProps["items"] = [
    {
      key: "doc-group",
      icon: <BookOutlined />,
      label: "Tài liệu",
      children: documents.map((doc) => ({
        key: `doc-${doc.key}`,
        icon: <FileTextOutlined />,
        label: doc.title,
      })),
    },
    {
      key: "ml-group",
      icon: <ExperimentOutlined />,
      label: "Công cụ Machine Learning",
      children: [
        {
          key: "ml-training",
          icon: <ThunderboltOutlined />,
          label: "Huấn luyện CNN Model",
        },
        {
          key: "ml-predict",
          icon: <RocketOutlined />,
          label: "Dự đoán chữ số",
        },
      ],
    },
  ];

  // Xử lý chuyển hướng khi click Menu
  const handleMenuClick = ({ key }: { key: string }) => {
    if (key === "ml-training") {
      router.push("/ml/training");
    } else if (key === "ml-predict") {
      router.push("/ml/predict");
    } else if (key.startsWith("doc-")) {
      const slug = key.replace("doc-", "");
      router.push(`/doc/${slug}`);
    }
  };

  // Xác định tiêu đề hiển thị trên Header
  const getHeaderInfo = () => {
    if (selectedKey === "ml-training") {
      return { title: "Huấn luyện CNN Model", icon: <ThunderboltOutlined style={{ color: "var(--accent)" }} /> };
    }
    if (selectedKey === "ml-predict") {
      return { title: "Dự đoán chữ số", icon: <RocketOutlined style={{ color: "var(--success)" }} /> };
    }
    const doc = documents.find((d) => `doc-${d.key}` === selectedKey);
    return { title: doc ? doc.title : "Tài liệu", icon: <FileTextOutlined style={{ color: "#a855f7" }} /> };
  };

  const headerInfo = getHeaderInfo();

  return (
    <ConfigProvider
      theme={{
        algorithm: antTheme.darkAlgorithm,
        token: {
          colorPrimary: "#6366f1",
          borderRadius: 8,
          fontFamily: "'Inter', sans-serif",
        },
      }}
    >
      <Layout style={{ minHeight: "100vh" }}>
        {/* ─── Sidebar ─── */}
        <Sider
          width={280}
          collapsible
          collapsed={collapsed}
          onCollapse={setCollapsed}
          breakpoint="lg"
          style={{
            position: "fixed",
            left: 0,
            top: 0,
            bottom: 0,
            zIndex: 100,
            overflow: "auto",
          }}
        >
          {/* Logo */}
          <div className="sidebar-logo">
            <div className="logo-icon">
              <DeploymentUnitOutlined style={{ fontSize: 20 }} />
            </div>
            {!collapsed && (
              <div>
                <div className="logo-text">CNN MNIST Lab</div>
                <div className="logo-sub">Toán cho KHMT</div>
              </div>
            )}
          </div>

          {/* Navigation Menu */}
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            defaultOpenKeys={["doc-group", "ml-group"]}
            items={menuItems}
            onClick={handleMenuClick}
            style={{ marginTop: 8 }}
          />
        </Sider>

        {/* ─── Main Area ─── */}
        <Layout
          style={{
            marginLeft: collapsed ? 80 : 280,
            transition: "margin-left 0.2s",
            height: "100vh",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {/* Header */}
          <Header className="app-header">
            <div className="header-left">
              <div className="header-title">
                <span>{headerInfo.icon}</span>
                <span>{headerInfo.title}</span>
              </div>
              <div className="header-breadcrumb">
                CNN MNIST Lab → {headerInfo.title}
              </div>
            </div>

            <div className="header-right">
              <div className="header-status">
                <span className="dot" />
                <span>NumPy Engine Active</span>
              </div>
            </div>
          </Header>

          {/* Content */}
          <Content
            className="main-content"
            style={{
              overflowY: "auto",
              height: "calc(100vh - 80px)",
            }}
          >
            {children}
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}
