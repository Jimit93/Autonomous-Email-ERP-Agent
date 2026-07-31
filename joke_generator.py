"""
Random Joke Generator
Fetches jokes from an external API and displays them with a nice interface.
Supports multiple joke types and formats.
"""

import requests
import json
from typing import Dict, Optional, List
import random

# ===========================================================================
# JOKE API CONFIGURATIONS
# ===========================================================================

class JokeAPI:
    """Base class for joke API interactions"""
    
    @staticmethod
    def get_joke_from_api() -> Optional[Dict]:
        """Fetch a random joke from JokeAPI (https://jokeapi.dev)"""
        try:
            # JokeAPI - Free joke API with various categories
            url = "https://v2.jokeapi.dev/joke/Any"
            params = {
                "format": "json",
                "safe-mode": True  # Family-friendly jokes
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("type") == "twopart":
                return {
                    "joke": data.get("setup"),
                    "punchline": data.get("delivery"),
                    "category": data.get("category", "General"),
                    "type": "two-part",
                    "source": "JokeAPI"
                }
            else:
                return {
                    "joke": data.get("joke"),
                    "punchline": None,
                    "category": data.get("category", "General"),
                    "type": "single",
                    "source": "JokeAPI"
                }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from JokeAPI: {e}")
            return None


class ProgrammingJokeAPI:
    """Fetch programming jokes"""
    
    @staticmethod
    def get_programming_joke() -> Optional[Dict]:
        """Fetch a programming joke from Official Joke API"""
        try:
            url = "https://official-joke-api.appspot.com/random_joke"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return {
                "joke": data.get("setup"),
                "punchline": data.get("punchline"),
                "category": data.get("type", "Programming"),
                "type": "two-part",
                "source": "Official Joke API"
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching programming joke: {e}")
            return None


class ChuckNorrisJokeAPI:
    """Fetch Chuck Norris jokes"""
    
    @staticmethod
    def get_chuck_norris_joke() -> Optional[Dict]:
        """Fetch a Chuck Norris joke"""
        try:
            url = "https://api.chucknorris.io/jokes/random"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return {
                "joke": data.get("value"),
                "punchline": None,
                "category": "Chuck Norris",
                "type": "single",
                "source": "Chuck Norris API"
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Chuck Norris joke: {e}")
            return None


class DadJokeAPI:
    """Fetch Dad jokes"""
    
    @staticmethod
    def get_dad_joke() -> Optional[Dict]:
        """Fetch a dad joke"""
        try:
            url = "https://icanhazdadjoke.com/random"
            headers = {"Accept": "application/json"}
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return {
                "joke": data.get("joke"),
                "punchline": None,
                "category": "Dad Joke",
                "type": "single",
                "source": "Dad Joke API"
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching dad joke: {e}")
            return None


# ===========================================================================
# JOKE GENERATOR CLASS
# ===========================================================================

class JokeGenerator:
    """Main Joke Generator with multiple sources"""
    
    def __init__(self):
        self.joke_sources = {
            "general": JokeAPI.get_joke_from_api,
            "programming": ProgrammingJokeAPI.get_programming_joke,
            "chuck_norris": ChuckNorrisJokeAPI.get_chuck_norris_joke,
            "dad": DadJokeAPI.get_dad_joke
        }
        self.joke_history: List[Dict] = []
    
    def get_random_joke(self, category: str = "random") -> Optional[Dict]:
        """
        Get a random joke from any source.
        
        Args:
            category: Type of joke ('general', 'programming', 'chuck_norris', 'dad', or 'random')
        
        Returns:
            Dictionary with joke data or None if failed
        """
        if category == "random":
            category = random.choice(list(self.joke_sources.keys()))
        
        if category not in self.joke_sources:
            print(f"Category '{category}' not found. Available: {list(self.joke_sources.keys())}")
            return None
        
        joke = self.joke_sources[category]()
        
        if joke:
            self.joke_history.append(joke)
        
        return joke
    
    def get_multiple_jokes(self, count: int = 5, category: str = "random") -> List[Dict]:
        """
        Get multiple random jokes.
        
        Args:
            count: Number of jokes to fetch
            category: Type of joke
        
        Returns:
            List of joke dictionaries
        """
        jokes = []
        for _ in range(count):
            joke = self.get_random_joke(category)
            if joke:
                jokes.append(joke)
        return jokes
    
    def display_joke(self, joke: Dict) -> None:
        """Display a joke in a formatted way"""
        if not joke:
            print("❌ Could not fetch a joke. Please try again!")
            return
        
        print("\n" + "="*70)
        print(f"📚 Category: {joke.get('category', 'Unknown')}")
        print(f"🔗 Source: {joke.get('source', 'Unknown')}")
        print("="*70)
        
        if joke.get("type") == "two-part":
            print(f"\n😄 Setup: {joke.get('joke')}")
            print(f"\n😂 Punchline: {joke.get('punchline')}")
        else:
            print(f"\n😄 {joke.get('joke')}")
        
        print("\n" + "="*70 + "\n")
    
    def display_multiple_jokes(self, jokes: List[Dict]) -> None:
        """Display multiple jokes"""
        print(f"\n🎭 Got {len(jokes)} jokes for you!\n")
        for i, joke in enumerate(jokes, 1):
            print(f"--- Joke {i} ---")
            self.display_joke(joke)
    
    def save_jokes_to_file(self, filename: str = "jokes.json") -> None:
        """Save joke history to a JSON file"""
        if not self.joke_history:
            print("No jokes to save!")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.joke_history, f, indent=2)
        print(f"✅ Saved {len(self.joke_history)} jokes to {filename}")
    
    def display_history(self) -> None:
        """Display all jokes in history"""
        if not self.joke_history:
            print("No jokes in history yet!")
            return
        
        print(f"\n📋 Joke History ({len(self.joke_history)} jokes)\n")
        for i, joke in enumerate(self.joke_history, 1):
            print(f"{i}. [{joke.get('category')}] {joke.get('joke')[:60]}...")


# ===========================================================================
# DEMO FUNCTIONS
# ===========================================================================

def demo_single_joke():
    """Demo: Fetch and display a single random joke"""
    print("\n🎭 RANDOM JOKE GENERATOR 🎭\n")
    
    generator = JokeGenerator()
    
    print("Fetching a random joke from any source...")
    joke = generator.get_random_joke()
    generator.display_joke(joke)


def demo_category_jokes():
    """Demo: Fetch jokes from specific categories"""
    print("\n🎭 JOKES BY CATEGORY 🎭\n")
    
    generator = JokeGenerator()
    categories = ["general", "programming", "chuck_norris", "dad"]
    
    for category in categories:
        print(f"\n--- {category.upper()} JOKE ---")
        joke = generator.get_random_joke(category)
        generator.display_joke(joke)


def demo_interactive_menu():
    """Demo: Interactive menu for users"""
    print("\n" + "="*70)
    print("🎭 WELCOME TO RANDOM JOKE GENERATOR 🎭".center(70))
    print("="*70)
    
    generator = JokeGenerator()
    
    while True:
        print("\n📌 MENU:")
        print("1. Get a random joke")
        print("2. Get a programming joke")
        print("3. Get a Chuck Norris joke")
        print("4. Get a dad joke")
        print("5. Get multiple jokes (5)")
        print("6. View joke history")
        print("7. Save jokes to file")
        print("8. Exit")
        
        choice = input("\n👉 Enter your choice (1-8): ").strip()
        
        if choice == "1":
            joke = generator.get_random_joke("random")
            generator.display_joke(joke)
        
        elif choice == "2":
            joke = generator.get_random_joke("programming")
            generator.display_joke(joke)
        
        elif choice == "3":
            joke = generator.get_random_joke("chuck_norris")
            generator.display_joke(joke)
        
        elif choice == "4":
            joke = generator.get_random_joke("dad")
            generator.display_joke(joke)
        
        elif choice == "5":
            jokes = generator.get_multiple_jokes(count=5, category="random")
            generator.display_multiple_jokes(jokes)
        
        elif choice == "6":
            generator.display_history()
        
        elif choice == "7":
            filename = input("Enter filename (default: jokes.json): ").strip() or "jokes.json"
            generator.save_jokes_to_file(filename)
        
        elif choice == "8":
            print("\n👋 Thanks for using Joke Generator! Goodbye!\n")
            break
        
        else:
            print("❌ Invalid choice. Please try again!")


if __name__ == "__main__":
    demo_interactive_menu()