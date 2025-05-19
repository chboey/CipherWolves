from faker import Faker

def generate_keywords(num_keywords):
    """
    Generate random English keywords using Faker.
    
    Args:
        num_keywords (int): Number of keywords to generate
        word_type (str): Type of words to generate ('word')
    
    Returns:
        list: List of generated keywords
    
    Example:
        >>> generate_keywords(10)
        ['apple', 'banana', 'orange', 'pear', 'pineapple', 'strawberry', 'watermelon', 'grape', 'mango', 'kiwi']
    """
    fake = Faker()
    keywords = []
    
    for _ in range(num_keywords):
        keywords.append(fake.word())
    
    return keywords

