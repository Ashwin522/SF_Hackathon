"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Send, Paperclip, Smile, MoreVertical, Bot, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

interface Message {
  id: number
  sender: "user" | "ai" 
  name: string
  content: string
  timestamp: string
  avatar?: string
}

const initialMessages: Message[] = [

]

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [inputValue, setInputValue] = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])


  // const sendMessageToBackend = async (content: string): Promise<Message> => {
  //   const response = await fetch("https://your-backend.com/chat", {
  //     method: "POST",
  //     headers: { "Content-Type": "application/json" },
  //     body: JSON.stringify({ content }),
  //   });
  //   const data = await response.json();
  //   return data as Message;
  // };

  const sendMessageToBackend = async (content: string): Promise<Message> => {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    if (!res.ok) throw new Error("LLM request failed");
    return res.json();
  };

  const nowTime = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });


  const handleSendMessage = async () => {
    const messageContent = inputValue.trim();
    if (!messageContent) return;

    const newMessage: Message = {
      id: messages.length + 1,
      sender: "user",
      name: "You",
      content: messageContent,
      timestamp: nowTime(),
    };

    setMessages((prev) => [...prev, newMessage]);
    setInputValue("");

    try {
      const aiResponse = await sendMessageToBackend(messageContent);
      setMessages((prev) => [...prev, aiResponse]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          id: prev.length + 2,
          sender: "ai",
          name: "AI Assistant",
          content: "Sorry, I couldn't process your message.",
          timestamp: nowTime(),
        },
      ]);
    }
  };

  // const handleSendMessage = async () => {
  //   if (!inputValue.trim()) return

  //   const newMessage: Message = {
  //     id: messages.length + 1,
  //     sender: "user",
  //     name: "You",
  //     content: inputValue,
  //     timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  //   }

  //   setMessages([...messages, newMessage])
  //   setInputValue("")

  //   // Simulate AI response
  //   // setTimeout(() => {
  //   //   const aiResponse: Message = {
  //   //     id: messages.length + 2,
  //   //     sender: "ai",
  //   //     name: "AI Assistant",
  //   //     content: "I'll analyze that for you. Let me pull up the relevant data from the results panel...",
  //   //     timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  //   //   }
  //   //   setMessages((prev) => [...prev, aiResponse])
  //   // }, 1000)

  //      try {
  //        const aiResponse = await sendMessageToBackend(inputValue);
  //        setMessages((prev) => [...prev, aiResponse]);
  //      } catch (error) {
  //        console.error(error);
  //        // Optional: show error message in chat
  //        setMessages((prev) => [
  //          ...prev,
  //          {
  //            id: prev.length + 2,
  //            sender: "ai",
  //            name: "AI Assistant",
  //            content: "Sorry, I couldn't process your message.",
  //            timestamp: nowTime(),
  //          },
  //        ]);
  //      }

  // }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="h-[95%] flex flex-col bg-card">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div>
          <h2 className="font-semibold text-foreground">Discussion</h2>
          <p className="text-xs text-muted-foreground">Chat about the results</p>
        </div>
        <Button variant="ghost" size="icon">
          <MoreVertical className="h-5 w-5" />
        </Button>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 px-4 py-4" ref={scrollRef}>
        <div className="space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex gap-3 ${message.sender === "user" ? "flex-row-reverse" : ""}`}>
              <Avatar className="h-8 w-8 shrink-0">
                {message.sender === "ai" ? (
                  <AvatarFallback className="bg-primary text-primary-foreground">
                    <Bot className="h-4 w-4" />
                  </AvatarFallback>
                ) : message.sender === "user" ? (
                  <AvatarFallback className="bg-accent text-accent-foreground">
                    <User className="h-4 w-4" />
                  </AvatarFallback>
                ) : (
                  <>
                    <AvatarImage src={message.avatar || "/placeholder.svg"} />
                    <AvatarFallback>{message.name.charAt(0)}</AvatarFallback>
                  </>
                )}
              </Avatar>
              <div className={`flex flex-col max-w-[80%] ${message.sender === "user" ? "items-end" : "items-start"}`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-foreground">{message.name}</span>
                  <span className="text-xs text-muted-foreground">{message.timestamp}</span>
                </div>
                <div
                  className={`rounded-lg px-3 py-2 text-sm ${
                    message.sender === "user"
                      ? "bg-primary text-primary-foreground"
                      : message.sender === "ai"
                        ? "bg-secondary border border-primary/20 text-foreground"
                        : "bg-secondary text-foreground"
                  }`}
                >
                  {message.content}
                </div>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center gap-2 bg-secondary rounded-lg px-3 py-2">
          {/* <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
            <Paperclip className="h-4 w-4" />
          </Button> */}
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Type a message..."
            className="flex-1 border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 px-0"
          />
          <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
            <Smile className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            className="h-8 w-8 rounded-full"
            onClick={handleSendMessage}
            disabled={!inputValue.trim()}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        {/* <p className="text-xs text-muted-foreground mt-2 text-center">
          AI can analyze results and answer questions about the data
        </p> */}
      </div>
    </div>
  )
}
