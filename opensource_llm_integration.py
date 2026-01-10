#!/usr/bin/env python3
"""
Open Source LLM Integration for Basketball Tactical Analysis
Uses Together AI API with open-source models (Llama, Mistral, etc.)
"""

import requests
import os
import json


class OpenSourceLLM:
    """Integration with open-source LLM APIs"""
    
    def __init__(self, provider="together"):
        """
        Initialize LLM client
        provider: "together", "huggingface", or "groq"
        """
        self.provider = provider
        self.api_key = None
        self.api_url = None
        self.model_name = None
        
        self._setup_provider()
    
    def _setup_provider(self):
        """Setup API configuration based on provider"""
        if self.provider == "together":
            # Together AI - Free tier: https://together.ai
            self.api_key = os.getenv('TOGETHER_API_KEY')
            self.api_url = "https://api.together.xyz/v1/chat/completions"
            self.model_name = "meta-llama/Llama-3-8b-chat-hf"  # Free, fast, good quality
            
        elif self.provider == "groq":
            # Groq - Free tier: https://groq.com
            self.api_key = os.getenv('GROQ_API_KEY')
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model_name = "llama-3.1-8b-instant"  # Extremely fast
            
        elif self.provider == "huggingface":
            # Hugging Face Inference API
            self.api_key = os.getenv('HUGGINGFACE_API_KEY')
            self.api_url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
            self.model_name = "Meta-Llama-3-8B-Instruct"
    
    def generate_tactical_analysis(self, attacking_positions, defending_positions, ball_pos, possession):
        """
        Generate tactical analysis using open-source LLM
        """
        if not self.api_key:
            raise ValueError(f"API key not set for {self.provider}. Set environment variable: {self._get_env_var_name()}")
        
        # Format positions for LLM
        prompt = self._format_tactical_prompt(attacking_positions, defending_positions, ball_pos, possession)
        
        # Call API
        try:
            response = self._call_api(prompt)
            return response
        except Exception as e:
            print(f"LLM API Error: {e}")
            raise
    
    def _get_env_var_name(self):
        """Get environment variable name for current provider"""
        if self.provider == "together":
            return "TOGETHER_API_KEY"
        elif self.provider == "groq":
            return "GROQ_API_KEY"
        elif self.provider == "huggingface":
            return "HUGGINGFACE_API_KEY"
    
    def _format_tactical_prompt(self, attacking_positions, defending_positions, ball_pos, possession):
        """Format basketball game state into LLM prompt"""
        team_name = "Team 1" if possession == 0 else "Team 2"
        
        # Format attacking positions
        atk_str = "\n".join([f"  P{pid}: X={x:.1f}, Y={y:.1f}" 
                             for pid, (x, y) in sorted(attacking_positions.items())])
        
        # Format defending positions
        def_str = "\n".join([f"  D{pid}: X={x:.1f}, Y={y:.1f}" 
                             for pid, (x, y) in sorted(defending_positions.items())])
        
        prompt = f"""You are a professional basketball tactical analyst. Analyze this defensive formation:

GAME STATE:
Ball Position: X={ball_pos:.1f}
Attacking Team: {team_name}
Court: X-axis 0-100 (length), Y-axis 0-50 (width)

ATTACKING POSITIONS:
{atk_str}

DEFENDING POSITIONS:
{def_str}

TASK: Provide tactical analysis with:
1. DEFENSIVE GAPS: Identify spacing weaknesses (gaps > 10 units between defenders)
2. ISOLATED DEFENDERS: Players >15 units from teammates
3. COUNTER-STRATEGY: Recommend specific attacking moves
4. OPTIMAL POSITIONS: Where should attacking players move (give X,Y coordinates)

Keep response under 200 words, be specific and actionable."""
        
        return prompt
    
    def _call_api(self, prompt):
        """Call the LLM API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.provider in ["together", "groq"]:
            # OpenAI-compatible chat format
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a professional basketball tactical analyst."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['message']['content']
        
        elif self.provider == "huggingface":
            # Hugging Face format
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', str(result))
            return str(result)


def test_llm_providers():
    """Test different open-source LLM providers"""
    print("="*80)
    print("TESTING OPEN-SOURCE LLM PROVIDERS")
    print("="*80 + "\n")
    
    # Sample game state
    attacking_positions = {1: (10, 25), 2: (25, 15), 3: (25, 35), 4: (45, 20), 5: (45, 30)}
    defending_positions = {1: (15, 25), 2: (30, 18), 3: (30, 32), 4: (50, 22), 5: (50, 28)}
    ball_pos = 10
    possession = 0
    
    providers = ["together", "groq", "huggingface"]
    
    for provider in providers:
        print(f"\n{'─'*80}")
        print(f"Testing: {provider.upper()}")
        print(f"{'─'*80}")
        
        try:
            llm = OpenSourceLLM(provider=provider)
            print(f"Model: {llm.model_name}")
            print(f"API Key: {'✓ Set' if llm.api_key else '✗ Not Set'}")
            
            if not llm.api_key:
                print(f"⚠ Skipping - Set {llm._get_env_var_name()} environment variable")
                continue
            
            print("\nCalling API...")
            response = llm.generate_tactical_analysis(
                attacking_positions, defending_positions, ball_pos, possession
            )
            
            print("\n✓ SUCCESS - Response:")
            print("-" * 80)
            print(response[:500])  # Print first 500 chars
            print("-" * 80)
            
        except Exception as e:
            print(f"\n✗ FAILED: {e}")


def setup_instructions():
    """Print setup instructions for each provider"""
    print("\n" + "="*80)
    print("SETUP INSTRUCTIONS - Choose ONE provider")
    print("="*80 + "\n")
    
    print("OPTION 1: Together AI (Recommended - Free tier available)")
    print("-" * 80)
    print("1. Sign up at: https://together.ai")
    print("2. Get API key from: https://api.together.xyz/settings/api-keys")
    print("3. Set environment variable:")
    print("   export TOGETHER_API_KEY='your-key-here'")
    print("   Models: Llama-3-8b, Mistral-7B, many others")
    print()
    
    print("OPTION 2: Groq (Fastest - Free tier available)")
    print("-" * 80)
    print("1. Sign up at: https://console.groq.com")
    print("2. Get API key from: https://console.groq.com/keys")
    print("3. Set environment variable:")
    print("   export GROQ_API_KEY='your-key-here'")
    print("   Models: Llama-3.1-8b (extremely fast), Mixtral-8x7b")
    print()
    
    print("OPTION 3: Hugging Face Inference API")
    print("-" * 80)
    print("1. Sign up at: https://huggingface.co")
    print("2. Get token from: https://huggingface.co/settings/tokens")
    print("3. Set environment variable:")
    print("   export HUGGINGFACE_API_KEY='your-token-here'")
    print("   Models: Meta-Llama-3-8B, Mistral-7B, many others")
    print()
    
    print("="*80)
    print("After setting up, run: python opensource_llm_integration.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_instructions()
    else:
        test_llm_providers()
