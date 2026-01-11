// "use client"

// import { useState } from "react"
// import { VideoPanel } from "@/components/video-panel"
// import { ResultsPanel } from "@/components/results-panel"
// import { ChatPanel } from "@/components/chat-panel"

// export default function VideoChatPage() {
//   const [isResultsExpanded, setIsResultsExpanded] = useState(false)
//   const [videoUrl, setVideoUrl] = useState<string | null>(null)

//   return (
//     <div className="flex h-screen bg-background">
//       {/* Left Section - Video + Results */}
//       <div className="flex flex-col w-[65%]">
//         <div className={`transition-all duration-300 overflow-hidden ${isResultsExpanded ? "h-0" : "h-[55%]"}`}>
//           <VideoPanel videoUrl={videoUrl} onVideoUpload={setVideoUrl} />
//         </div>

//         <div className={`transition-all duration-300 ${isResultsExpanded ? "h-full" : "h-[45%]"}`}>
//           <ResultsPanel
//             isExpanded={isResultsExpanded}
//             onToggleExpand={() => setIsResultsExpanded(!isResultsExpanded)}
//           />
//         </div>
//       </div>

//       {/* Right Section - Chat (always visible) */}
//       <div className="w-[35%]  border-l border-border">
//         <ChatPanel />
//       </div>
//     </div>
//   )
// }


"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";

export default function LandingPage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground relative overflow-hidden">
      {/* Animated background grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px]" />

      {/* Gradient orbs */}
      <div className="absolute top-1/4 -left-32 w-96 h-96 bg-primary/20 rounded-full blur-[128px] animate-pulse" />
      <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-primary/10 rounded-full blur-[128px] animate-pulse delay-1000" />

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center px-6">
        <div
          className={`text-center transition-all duration-1000 ${
            mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
          }`}
        >
          {/* Logo mark */}
          <div className="mb-12 flex justify-center">
            <div className="relative">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center shadow-2xl shadow-primary/30">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  className="w-10 h-10 text-primary-foreground"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 2C12 2 12 12 12 12C12 12 22 12 22 12" />
                  <path d="M12 22C12 22 12 12 12 12C12 12 2 12 2 12" />
                  <path d="M4.93 4.93C4.93 4.93 12 12 12 12C12 12 19.07 19.07 19.07 19.07" />
                </svg>
              </div>
              <div className="absolute -inset-1 bg-gradient-to-br from-primary to-primary/60 rounded-2xl blur-xl opacity-50 -z-10" />
            </div>
          </div>

          {/* Title */}
          <h1 className="text-6xl md:text-8xl font-bold tracking-tight mb-6 text-balance">
            <span className="bg-gradient-to-b from-foreground to-foreground/70 bg-clip-text text-transparent">
              CourtVision
            </span>
          </h1>

          {/* Tagline */}
          <p className="text-muted-foreground text-xl md:text-2xl mb-16 font-light tracking-wide">
            AI-powered basketball defense analysis
          </p>

          {/* CTA Button */}
          <Link href="/analyze">
            <button className="group relative inline-flex items-center gap-3 px-8 py-4 bg-foreground text-background rounded-full font-medium text-lg transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:shadow-foreground/20">
              <span>Start Analyzing</span>
              <ArrowRight className="w-5 h-5 transition-transform duration-300 group-hover:translate-x-1" />
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-primary/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            </button>
          </Link>
        </div>

        {/* Bottom hint */}
        <div
          className={`absolute bottom-8 text-muted-foreground/50 text-sm transition-all duration-1000 delay-500 ${
            mounted ? "opacity-100" : "opacity-0"
          }`}
        >
          Upload footage. Get insights.
        </div>
      </div>
    </div>
  );
}
