
import yaml

# open, parse, return
def load_policy(path="agent_policy.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
    



def is_allowed(policy, agent, resource, action):
    # Step 1 — agent side: find the agent's entry for this resource
    agent_allowed = policy["agents"][agent]["allowed"]  # a list of {resource, scope} dicts
    agent_scope = next(
        (e["scope"] for e in agent_allowed 
         if e["resource"] == resource),  
         None,                            # default if no match
    )
    if agent_scope is None:
        return None, "agent has no access to this resource"         # agent isn't permitted this resource at all
    

    # Step 2 — human side: find the human's entry for this resource
    owner = policy["agents"][agent]["owner"]     # richard@barrikade.ai
    human_allowed = policy["humans"][owner]["allowed"] # NOW it's the list of {resource, scope} dicts
    human_scope = next(
        (e["scope"] for e in human_allowed
         if e["resource"] == resource),
         None,
    )
    if human_scope is None:
        return None, "human owner lacks this resource"                # human isn't permitted -> intersection empty
    
    # Step 3 — both sides had it: return the agent's (narrower) scope
    if action in agent_scope:
        return agent_scope, "ok"
    return None, f"agent scope '{agent_scope}' does not cover '{action}'"  # scope doen't cover this action -> deny




if __name__ == "__main__":
    p = load_policy()
    print(is_allowed(p, "notion-bugfinder", "github", "read"))   # -> repo:read
    print(is_allowed(p, "notion-bugfinder", "github", "write"))  # -> None (human has write, agent doesn't...)
    print(is_allowed(p, "notion-bugfinder", "stripe", "read"))   # -> None

    print(is_allowed(p, "deploy-bot", "github", "write"))   # 
