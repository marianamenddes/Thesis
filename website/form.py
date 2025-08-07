from django import forms
from .models import Patient # Ensure your Patient model is imported
# from django.contrib.auth.models import User # Not necessary for this patient form


class PatientForm(forms.ModelForm):
    # Choices for the 'sex' field (previously 'gender')
    # Use 'Sex' instead of 'Gender' for medical clarity, if applicable.
    SEX_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
        # Consider adding 'Other' or 'Unspecified' if needed for broader inclusivity
    ]
    sex = forms.ChoiceField(
        choices=SEX_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}), # Add CSS class for styling
        label="Sex" # Label to be displayed on the form
    )

    # Phone Number: CharField, with a placeholder and help text for guidance.
    # A custom clean method is added for server-side numeric-only validation.
    phone_number = forms.CharField(
        max_length=20,
        required=False, # Set to True if the phone number is mandatory
        widget=forms.TextInput(attrs={'placeholder': 'e.g., 912345678'}),
        help_text="Enter numbers only, without spaces or symbols."
    )

    # Weight and Height: DecimalField for precision.
    # The HTML5 'NumberInput' widget is used, which typically expects a dot '.' for decimals.
    # Units (kg, cm) are displayed in the HTML template next to the input field.
    weight = forms.DecimalField(
        max_digits=5, # Total number of digits allowed
        decimal_places=2, # Number of decimal places
        widget=forms.NumberInput(attrs={
            'step': '0.01', # Allows for decimal input
            'placeholder': 'e.g., 75.5',
            'min': '0.01'
        }),
        label="Weight"
    )

    height = forms.DecimalField(
        max_digits=5,
        decimal_places=2,
        widget=forms.NumberInput(attrs={
            'step': '0.01',
            'placeholder': 'e.g., 170.5',
            'min': '0.01'
        }),
        label="Height"
    )

    class Meta:
        model = Patient # Ensure 'Patient' is the name of your model
        # Ensure the fields here match your model fields and the fields defined above.
        fields = ['name', 'age', 'sex', 'weight', 'height', 'phone_number', 'dominant_leg_side']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Full Name'}),
            'age': forms.NumberInput(attrs={'placeholder': 'Age in years', 'min': '1'}),
            # 'dominant_leg_side' can use a default widget or a RadioSelect if it has few options.
            # For example: 'dominant_leg_side': forms.RadioSelect(),
        }

    # Custom validation for the phone number to ensure it contains only digits
    def clean_phone_number(self):
        phone_number = self.cleaned_data.get('phone_number')
        if phone_number:
            # Remove any non-digit characters if you expect them (e.g., spaces, dashes)
            clean_phone_number = ''.join(filter(str.isdigit, phone_number))
            if not clean_phone_number.isdigit(): # Check if after cleaning, it's still not all digits
                raise forms.ValidationError("Phone number must contain only digits.")
            return clean_phone_number
        return phone_number

    # Nova validação para 'age' ser positivo
    def clean_age(self):
        age = self.cleaned_data.get('age')
        if age is not None and age <= 0:
            raise forms.ValidationError("Age must be a positive number.")
        return age

    # Nova validação para 'height' ser positivo
    def clean_height(self):
        height = self.cleaned_data.get('height')
        if height is not None and height <= 0:
            raise forms.ValidationError("Height must be a positive number.")
        return height

    # Nova validação para 'weight' ser positivo
    def clean_weight(self):
        weight = self.cleaned_data.get('weight')
        if weight is not None and weight <= 0:
            raise forms.ValidationError("Weight must be a positive number.")
        return weight