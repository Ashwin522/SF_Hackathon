"use client"

import { useState } from "react"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"
import { VideoPanel } from "@/components/video-panel"
import { ResultsPanel } from "@/components/results-panel"
import { ChatPanel } from "@/components/chat-panel"

export default function AnalyzePage() {
  const [isResultsExpanded, setIsResultsExpanded] = useState(false)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* <header className="h-14 flex items-center px-4 border-b border-border bg-background/80 backdrop-blur-md shrink-0">
        <Link
          href="/"
          className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-muted-foreground hover:text-foreground hover:bg-white/10 hover:border-white/20 transition-all duration-300 group"
        >
          <ArrowLeft className="w-4 h-4 transition-transform group-hover:-translate-x-1" />
          <span className="text-sm font-medium">Back</span>
        </Link>
        <h1 className="ml-4 text-lg font-semibold text-foreground">CourtVision AI</h1>
      </header> */}

      <div className="flex flex-1 overflow-hidden">
        {/* Left Section - Video + Results */}
        <div className="flex flex-col w-[65%]">
          <div className={`transition-all duration-300 overflow-hidden ${isResultsExpanded ? "h-0" : "h-[55%]"}`}>
            <VideoPanel videoUrl={videoUrl} onVideoUpload={setVideoUrl} />
          </div>

          <div className={`transition-all duration-300 ${isResultsExpanded ? "h-full" : "h-[45%]"}`}>
            <ResultsPanel
              isExpanded={isResultsExpanded}
              onToggleExpand={() => setIsResultsExpanded(!isResultsExpanded)}
            />
          </div>
        </div>

        {/* Right Section - Chat (always visible) */}
        <div className="w-[35%] border-l border-border">
          <ChatPanel />
        </div>
      </div>
    </div>
  )
}
