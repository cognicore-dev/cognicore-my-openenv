# CogniCore Extension Demo Scenario

This document outlines a demonstration scenario to showcase the value of the CogniCore Memory Extension in Claude Desktop.

## Preparation
1. Install the `.mcpb` bundle into Claude Desktop.
2. Ensure you are starting with a fresh chat.

## The Demo

### Step 1: Storing a Preference
**User:** 
> I'm building a new backend service. Please remember for all future conversations that I always use FastAPI for my web frameworks and strictly prefer `pytest` over `unittest`.

**Claude:**
*(Claude will call the `cognicore_remember` tool)*
> I've saved that preference to memory. I will use FastAPI and pytest for your backend services going forward.

### Step 2: Storing a Fact
**User:**
> Also remember that the staging database URL is `postgresql://staging:5432/main_db`.

**Claude:**
*(Claude calls `cognicore_remember`)*
> Got it. I've stored the staging database URL.

### Step 3: Proving Persistent Memory (The "Aha!" Moment)
*(Start a completely NEW chat in Claude Desktop. The context window is completely empty.)*

**User:**
> I need to write a quick health check endpoint for my backend, and a test for it. Please use the staging database URL in the endpoint code as a placeholder.

**Claude:**
*(Because Claude lacks the context, it will likely call `cognicore_recall` searching for "backend framework", "testing preference", or "staging database URL" based on its system prompt directing it to use memory tools).*
> *(Retrieves FastAPI, pytest, and the postgresql URL)*
> Here is a FastAPI health check endpoint and a pytest script, using your staging database URL...

### Step 4: Updating Memory
**User:**
> Actually, we migrated the database. Please forget the old staging URL and remember the new one is `postgresql://db-new.staging:5432/core`.

**Claude:**
*(Claude calls `cognicore_forget` and `cognicore_remember`)*
> I've updated the memory. The old URL has been deleted and the new one is saved.
