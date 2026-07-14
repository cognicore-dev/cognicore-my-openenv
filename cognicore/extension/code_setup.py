import os
import subprocess
import sys

def main():
    print("Setting up CogniCore Memory for Claude Code...")
    print("This will add the 'cognicore-memory' MCP server to your Claude Code configuration.")
    print("Make sure you have Claude Code installed (`npm install -g @anthropic-ai/claude-code`).\n")
    
    # We want to run: claude mcp add cognicore-memory uvx cognicore-env[extension] cognicore-extension
    # Or since we are installing it in the current env, we could use the current uvx if available, 
    # but the simplest robust way is exactly what the user requested for one-command integration.
    
    cmd = [
        "claude", 
        "mcp", 
        "add", 
        "cognicore-memory", 
        "uvx", 
        "cognicore-env[extension]", 
        "cognicore-extension"
    ]
    
    try:
        # We use shell=True on Windows if `claude` is a cmd/ps1 script from npm
        is_win = sys.platform.startswith("win")
        result = subprocess.run(cmd, check=True, shell=is_win)
        
        print("\n✅ Success! CogniCore Memory is now integrated with Claude Code.")
        print("When you run `claude`, it will have access to persistent global and project-scoped memory.")
        print("Try asking it to remember a project fact, then start a new session to see it recalled!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to add MCP server. Claude Code CLI might not be installed or configured correctly.")
        print(f"Error details: {e}")
        print("\nYou can try adding it manually by running:")
        print(" ".join(cmd))
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Could not find the 'claude' command.")
        print("Please ensure Claude Code is installed and in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    main()
