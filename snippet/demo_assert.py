def apply_discount(price, discount_percentage):
    # The discount percentage must be between 0 and 100.
    assert 0 <= discount_percentage <= 100, "Discount must be between 0 and 100."
    
    final_price = price * (1 - discount_percentage / 100)
    
    # The final price should never be more than the original price.
    assert final_price <= price, "Final price cannot be greater than original price."
    
    return final_price

# This will work fine
print(apply_discount(100, 10)) 

# This will raise an AssertionError
try:
    apply_discount(100, 150)
except AssertionError as e:
    print(f"Caught an error: {e}")