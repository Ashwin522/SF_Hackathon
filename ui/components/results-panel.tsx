"use client"
import { useState, useEffect } from "react";
import { ChevronUp, ChevronDown, TrendingUp, Target, AlertTriangle, Shield } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

interface ResultsPanelProps {
  isExpanded: boolean
  onToggleExpand: () => void
}

// const playerPositions = {
//   offense: [
//     { id: 1, x: 50, y: 25, label: "PG" },
//     { id: 2, x: 20, y: 35, label: "SG" },
//     { id: 3, x: 80, y: 35, label: "SF" },
//     { id: 4, x: 30, y: 60, label: "PF" },
//     { id: 5, x: 70, y: 60, label: "C" },
//   ],
//   defense: [
//     { id: 1, x: 48, y: 30, label: "1" },
//     { id: 2, x: 25, y: 40, label: "2" },
//     { id: 3, x: 75, y: 40, label: "3" },
//     { id: 4, x: 35, y: 55, label: "4" },
//     { id: 5, x: 65, y: 55, label: "5" },
//   ],
//   recommendedDefense: [
//     { id: 1, x: 50, y: 28, label: "1" },
//     { id: 2, x: 22, y: 38, label: "2" },
//     { id: 3, x: 78, y: 38, label: "3" },
//     { id: 4, x: 32, y: 58, label: "4" },
//     { id: 5, x: 68, y: 58, label: "5" },
//   ],
//   ball: { x: 50, y: 25 },
// }

const analysisText = {
  possession: {
    timestamp: "12.4s",
    event: "Paint Drive Occurred",
  },
  findings: [
    {
      type: "late_help",
      title: "Late Help Defense",
      description:
        "Help defender stayed glued to corner shooter instead of stunting to nail position. Arrived >0.5s later than recommended.",
      impact: "High",
    },
    {
      type: "missed_closeout",
      title: "Missed Closeout",
      description:
        "Corner shooter open distance >2.5m at catch. Defender failed to close out in time after ball rotation.",
      impact: "Medium",
    },
    {
      type: "rotation_error",
      title: "Rotation Correctness",
      description: "Weakside low man did not rotate to protect paint. Recommended stunt to nail for 0.6s then recover.",
      impact: "High",
    },
    {
      type: "assignment_error",
      title: "Assignment Error",
      description:
        "Distance of defender #4 to recommended location: 1.8m off optimal position during pick-and-roll coverage.",
      impact: "Medium",
    },
  ],
  metrics: {
    coverageError: "2.3m avg",
    paintProtection: "68%",
    closeoutQuality: "Low",
    assignmentAccuracy: "74%",
  },
  recommendation:
    "At 12.4s of possession, help defender should stunt to nail position for 0.6s then recover to corner. Weakside rotation needed to protect paint on drive.",
}

const getImpactColor = (impact: string) => {
  switch (impact) {
    case "High":
      return "text-red-400 border-red-400/30"
    case "Medium":
      return "text-yellow-400 border-yellow-400/30"
    case "Low":
      return "text-green-400 border-green-400/30"
    default:
      return "text-muted-foreground"
  }
}

// function BasketballCourt({ isExpanded }: { isExpanded: boolean }) {
//   return (
//     <div
//       className={`relative bg-[#1a1a2e] rounded-lg border border-border overflow-hidden ${isExpanded ? "h-64" : "h-48"}`}
//     >
//       <svg viewBox="0 0 100 85" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
//         {/* Court background */}
//         <rect x="0" y="0" width="100" height="85" fill="#2d4a22" />

//         {/* Court lines */}
//         <rect x="2" y="2" width="96" height="81" fill="none" stroke="#ffffff" strokeWidth="0.5" />

//         {/* Three-point arc */}
//         <path d="M 6 2 L 6 20 Q 6 65 50 75 Q 94 65 94 20 L 94 2" fill="none" stroke="#ffffff" strokeWidth="0.4" />

//         {/* Paint/Key */}
//         <rect x="31" y="2" width="38" height="45" fill="none" stroke="#ffffff" strokeWidth="0.4" />

//         {/* Free throw circle */}
//         <circle cx="50" cy="47" r="12" fill="none" stroke="#ffffff" strokeWidth="0.4" />

//         {/* Basket */}
//         <circle cx="50" cy="8" r="2" fill="none" stroke="#ff6b35" strokeWidth="0.6" />
//         <rect x="44" y="2" width="12" height="4" fill="none" stroke="#ffffff" strokeWidth="0.3" />

//         {/* Restricted area */}
//         <path d="M 42 2 Q 42 16 50 16 Q 58 16 58 2" fill="none" stroke="#ffffff" strokeWidth="0.3" />

//         {/* Recommended defense positions (dotted circles) */}
//         {playerPositions.recommendedDefense.map((player) => (
//           <circle
//             key={`rec-${player.id}`}
//             cx={player.x}
//             cy={player.y}
//             r="3.5"
//             fill="none"
//             stroke="#60a5fa"
//             strokeWidth="0.4"
//             strokeDasharray="1,1"
//             opacity="0.6"
//           />
//         ))}

//         {/* Offense players (red) */}
//         {playerPositions.offense.map((player) => (
//           <g key={`off-${player.id}`}>
//             <circle cx={player.x} cy={player.y} r="3" fill="#ef4444" stroke="#ffffff" strokeWidth="0.3" />
//             <text x={player.x} y={player.y + 1} textAnchor="middle" fill="#ffffff" fontSize="2.5" fontWeight="bold">
//               {player.label}
//             </text>
//           </g>
//         ))}

//         {/* Defense players (blue) */}
//         {playerPositions.defense.map((player) => (
//           <g key={`def-${player.id}`}>
//             <circle cx={player.x} cy={player.y} r="3" fill="#3b82f6" stroke="#ffffff" strokeWidth="0.3" />
//             <text x={player.x} y={player.y + 1} textAnchor="middle" fill="#ffffff" fontSize="2.5" fontWeight="bold">
//               {player.label}
//             </text>
//           </g>
//         ))}

//         {/* Ball */}
//         <circle
//           cx={playerPositions.ball.x}
//           cy={playerPositions.ball.y}
//           r="1.5"
//           fill="#f97316"
//           stroke="#ffffff"
//           strokeWidth="0.2"
//         />
//       </svg>

//       {/* Legend */}
//       <div className="absolute bottom-2 left-2 flex items-center gap-3 text-xs">
//         <div className="flex items-center gap-1">
//           <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
//           <span className="text-muted-foreground">Offense</span>
//         </div>
//         <div className="flex items-center gap-1">
//           <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
//           <span className="text-muted-foreground">Defense</span>
//         </div>
//         <div className="flex items-center gap-1">
//           <div className="w-2.5 h-2.5 rounded-full border border-blue-400 border-dashed" />
//           <span className="text-muted-foreground">Recommended</span>
//         </div>
//       </div>

//       {/* Timestamp */}
//       <div className="absolute top-2 right-2 bg-background/80 px-2 py-1 rounded text-xs text-foreground">
//         T = {analysisText.possession.timestamp}
//       </div>
//     </div>
//   )
// }


function BasketballCourt({ isExpanded }: { isExpanded: boolean }) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchCourtImage = async () => {
      setLoading(true);
      setError(false);

      try {
        const res = await fetch("/api/court-image"); // your API endpoint
        
        if (!res.ok) throw new Error("Failed to fetch image");
        const data = await res.json();
        // const data = "/basketball_court_grid.png"
        setImageUrl(data.url); // assume backend returns { url: "..." }
        // setImageUrl(data)
      } catch (err) {
        console.error(err);
        setError(true);
      } finally {
        setLoading(false);
      }
    };

    fetchCourtImage();
  }, []);

  return (
    <div
      className={`relative rounded-lg border border-border overflow-hidden bg-[#1a1a2e] flex items-center justify-center ${
        isExpanded ? "h-64" : "h-48"
      }`}
    >
      {loading && (
        <p className="text-sm text-muted-foreground">Loading court image...</p>
      )}
      {error && <p className="text-sm text-red-400">Failed to load image</p>}
      {!loading && !error && imageUrl && (
        <img
          src={imageUrl}
          alt="Basketball Court"
          className="w-full h-full object-contain"
        />
      )}
    </div>
  );
}


export function ResultsPanel({ isExpanded, onToggleExpand }: ResultsPanelProps) {
  return (
    <div className="h-full bg-card border-t border-border flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-3">
          <TrendingUp className="h-5 w-5 text-primary" />
          <h2 className="font-semibold text-foreground">Defense Analysis</h2>
          <Badge variant="secondary" className="text-xs">
            {analysisText.findings.length} findings
          </Badge>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggleExpand}
          className="gap-2"
        >
          {isExpanded ? (
            <>
              <ChevronDown className="h-4 w-4" />
              <span className="text-sm">Show Video</span>
            </>
          ) : (
            <>
              <ChevronUp className="h-4 w-4" />
              <span className="text-sm">Full View</span>
            </>
          )}
        </Button>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Target className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-medium text-foreground">
                Court Position Analysis
              </h3>
            </div>
            <BasketballCourt isExpanded={isExpanded} />
          </div>
          {isExpanded && (
            <>
              <div
                className={`grid gap-3 ${
                  isExpanded ? "grid-cols-4" : "grid-cols-2"
                }`}
              >
                <div className="bg-secondary/50 rounded-lg p-3 border border-border">
                  <div className="text-xs text-muted-foreground mb-1">
                    Coverage Error
                  </div>
                  <div className="text-lg font-semibold text-foreground">
                    {analysisText.metrics.coverageError}
                  </div>
                </div>
                <div className="bg-secondary/50 rounded-lg p-3 border border-border">
                  <div className="text-xs text-muted-foreground mb-1">
                    Paint Protection
                  </div>
                  <div className="text-lg font-semibold text-foreground">
                    {analysisText.metrics.paintProtection}
                  </div>
                </div>
                <div className="bg-secondary/50 rounded-lg p-3 border border-border">
                  <div className="text-xs text-muted-foreground mb-1">
                    Closeout Quality
                  </div>
                  <div className="text-lg font-semibold text-yellow-400">
                    {analysisText.metrics.closeoutQuality}
                  </div>
                </div>
                <div className="bg-secondary/50 rounded-lg p-3 border border-border">
                  <div className="text-xs text-muted-foreground mb-1">
                    Assignment Accuracy
                  </div>
                  <div className="text-lg font-semibold text-foreground">
                    {analysisText.metrics.assignmentAccuracy}
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-3">
                  <AlertTriangle className="h-4 w-4 text-yellow-400" />
                  <h3 className="text-sm font-medium text-foreground">
                    Key Findings
                  </h3>
                </div>
                <div
                  className={`grid gap-3 ${
                    isExpanded ? "grid-cols-2" : "grid-cols-1"
                  }`}
                >
                  {analysisText.findings.map((finding, index) => (
                    <div
                      key={index}
                      className="p-3 rounded-lg bg-secondary/50 border border-border hover:border-primary/50 transition-colors"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-medium text-foreground text-sm">
                          {finding.title}
                        </h4>
                        <Badge
                          variant="outline"
                          className={`text-xs ${getImpactColor(
                            finding.impact
                          )}`}
                        >
                          {finding.impact}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {finding.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-primary/10 border border-primary/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Shield className="h-4 w-4 text-primary" />
                  <h3 className="text-sm font-medium text-foreground">
                    Coach Recommendation
                  </h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  {analysisText.recommendation}
                </p>
              </div>
            </>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
