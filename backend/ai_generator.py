import anthropic
from typing import List, Optional, Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage:
- **Content search tool**: Use for questions about specific course content or detailed educational materials
- **Course outline tool**: Use when users ask for course outlines, course structure, lesson lists, or complete course overview
- **Sequential tool calling**: You can make up to 2 rounds of tool calls to gather comprehensive information
  - Round 1: Gather initial information (e.g., course outline, basic search)  
  - Round 2: Use results from round 1 to refine search or get additional details
  - Examples: Search course X lesson 4 → then search for courses covering that topic
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Course Outline Queries:
When users ask for course outlines, structure, or complete course information, use the course outline tool to provide:
- Course title and link
- Instructor information  
- Complete lesson list with numbers and titles
Format the response clearly showing all course components.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course content questions**: Use content search tool first, then answer
- **Course outline questions**: Use course outline tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum rounds for sequential tool calling (default 2)
            
        Returns:
            Generated response as string
            
        Raises:
            Exception: If API call fails or other errors occur
        """
        
        try:
            # Route to multi-round logic if tools and tool_manager are available
            if tools and tool_manager:
                return self._execute_tool_rounds(
                    query=query,
                    conversation_history=conversation_history,
                    tools=tools,
                    tool_manager=tool_manager,
                    max_rounds=max_rounds
                )
            
            # Fallback to single API call without tools (preserves backward compatibility)
            system_content = (
                f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
                if conversation_history 
                else self.SYSTEM_PROMPT
            )
            
            # Prepare API call parameters efficiently
            api_params = {
                **self.base_params,
                "messages": [{"role": "user", "content": query}],
                "system": system_content
            }
            
            logger.debug(f"Making single API call to {self.model} with query: {query[:100]}...")
            
            # Get response from Claude (no tools in this path)
            response = self.client.messages.create(**api_params)
            
            # Return direct response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                logger.warning("Received empty response from API")
                return "I apologize, but I received an empty response. Please try your query again."
                
        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            return "Authentication failed. Please check the API key configuration."
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            return "Rate limit exceeded. Please try again in a moment."
        except anthropic.APIError as e:
            logger.error(f"API error occurred: {e}")
            return f"API error occurred: {str(e)}. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {e}", exc_info=True)
            return f"An unexpected error occurred: {str(e)}. Please try again."
    
    def _execute_tool_rounds(self, query: str,
                           conversation_history: Optional[str] = None,
                           tools: Optional[List] = None,
                           tool_manager=None,
                           max_rounds: int = 2) -> str:
        """
        Execute multiple rounds of tool calling and reasoning.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of rounds (default 2)
            
        Returns:
            Final response after up to max_rounds of tool calling
        """
        
        try:
            # Build system content efficiently
            system_content = (
                f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
                if conversation_history 
                else self.SYSTEM_PROMPT
            )
            
            # Initialize conversation with user query
            messages = [{"role": "user", "content": query}]
            
            # Execute up to max_rounds of tool calling
            for round_num in range(1, max_rounds + 1):
                logger.debug(f"Starting tool round {round_num}/{max_rounds}")
                
                response_text, has_tool_use, updated_messages = self._execute_single_round(
                    messages=messages,
                    system_content=system_content,
                    tools=tools,
                    tool_manager=tool_manager
                )
                
                # Update messages for next round
                messages = updated_messages
                
                # Check termination conditions
                if not has_tool_use:
                    logger.debug(f"Natural termination after round {round_num} - no tool use")
                    return response_text
                
                # If we've hit max rounds but there's still tool use pending,
                # execute one more round to complete the tool interaction gracefully
                if round_num >= max_rounds:
                    if has_tool_use and round_num == max_rounds:
                        logger.debug(f"Max rounds reached but tool use pending - executing final round")
                        # Execute one final round to handle the pending tool use
                        final_response_text, final_has_tool_use, final_messages = self._execute_single_round(
                            messages=messages,
                            system_content=system_content,
                            tools=tools,
                            tool_manager=tool_manager
                        )
                        return final_response_text
                    else:
                        logger.debug(f"Max rounds ({max_rounds}) reached - terminating")
                        return response_text
            
            # Fallback (shouldn't reach here with current logic)
            return "I apologize, but I couldn't complete the task within the allowed rounds."
            
        except Exception as e:
            logger.error(f"Unexpected error in _execute_tool_rounds: {e}", exc_info=True)
            return f"An unexpected error occurred during multi-round processing: {str(e)}. Please try again."
    
    def _execute_single_round(self, messages: List[Dict], 
                            system_content: str,
                            tools: Optional[List] = None,
                            tool_manager=None) -> tuple[str, bool, List[Dict]]:
        """
        Execute a single round of API call + complete all tool execution within that round.
        
        Args:
            messages: Current conversation messages
            system_content: System prompt content
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Tuple of (response_text, has_tool_use_for_next_round, updated_messages)
        """
        
        current_messages = messages.copy()
        
        # Keep executing tools until we get a non-tool response
        while True:
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": current_messages,
                "system": system_content
            }
            
            # Add tools if available (key change - keep tools in every round)
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            logger.debug(f"Making API call within single round")
            
            # Get response from Claude
            response = self.client.messages.create(**api_params)
            
            # Add assistant response to messages
            current_messages.append({"role": "assistant", "content": response.content})
            
            # Check if tools were used
            if response.stop_reason == "tool_use" and tool_manager:
                # Execute all tools in this response
                tool_results = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        try:
                            logger.debug(f"Executing tool: {content_block.name} with input: {content_block.input}")
                            tool_result = tool_manager.execute_tool(
                                content_block.name, 
                                **content_block.input
                            )
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result
                            })
                            
                        except Exception as e:
                            logger.error(f"Tool execution failed for {content_block.name}: {e}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": f"Tool execution failed: {str(e)}"
                            })
                
                # Add tool results and continue the loop
                if tool_results:
                    current_messages.append({"role": "user", "content": tool_results})
                    continue  # Continue the loop to make another API call
                else:
                    # No tool results - treat as end of round
                    break
            else:
                # No tool use - end of round
                if response.content and len(response.content) > 0:
                    response_text = response.content[0].text
                    return response_text, False, current_messages
                else:
                    logger.warning("Received empty response from API")
                    response_text = "I apologize, but I received an empty response. Please try your query again."
                    current_messages.append({"role": "assistant", "content": response_text})
                    return response_text, False, current_messages
        
        # If we break out of the loop, it means we had tool use but something went wrong
        response_text = "I apologize, but I encountered an issue processing tools."
        return response_text, False, current_messages
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        This is the original method maintained for backward compatibility.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        try:
            # Start with existing messages
            messages = base_params["messages"].copy()
            
            # Add AI's tool use response
            messages.append({"role": "assistant", "content": initial_response.content})
            
            # Execute all tool calls and collect results
            tool_results = []
            for content_block in initial_response.content:
                if content_block.type == "tool_use":
                    try:
                        logger.debug(f"Executing tool: {content_block.name} with input: {content_block.input}")
                        tool_result = tool_manager.execute_tool(
                            content_block.name, 
                            **content_block.input
                        )
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result
                        })
                        
                    except Exception as e:
                        logger.error(f"Tool execution failed for {content_block.name}: {e}")
                        # Add error as tool result so AI can handle it
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution failed: {str(e)}"
                        })
            
            # Add tool results as single message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            
            # Prepare final API call without tools (for backward compatibility)
            final_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }
            
            logger.debug("Making final API call after tool execution")
            
            # Get final response
            final_response = self.client.messages.create(**final_params)
            
            if final_response.content and len(final_response.content) > 0:
                return final_response.content[0].text
            else:
                logger.warning("Received empty final response after tool execution")
                return "I apologize, but I couldn't generate a proper response after using the tools. Please try again."
                
        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication error during tool execution: {e}")
            return "Authentication failed during tool execution. Please check the API key configuration."
        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit exceeded during tool execution: {e}")
            return "Rate limit exceeded during tool execution. Please try again in a moment."
        except anthropic.APIError as e:
            logger.error(f"API error during tool execution: {e}")
            return f"API error during tool execution: {str(e)}. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error during tool execution: {e}", exc_info=True)
            return f"An unexpected error occurred during tool execution: {str(e)}. Please try again."