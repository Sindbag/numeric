from django import forms


class NumericForm(forms.Form):
    default_attrs = {
        'cols': 40,
        'rows': 3,
        'placeholder': 'Function or CSV',
        'class': 'form-control',
    }
    f = forms.CharField(widget=forms.Textarea(attrs=default_attrs),
                        help_text='f(x, z, S, B)')
    density = forms.CharField(widget=forms.Textarea(attrs=default_attrs),
                              help_text='p(w)')
    S = forms.CharField(widget=forms.Textarea(attrs=default_attrs),
                        help_text='S(t)')
    z = forms.CharField(widget=forms.Textarea(attrs=default_attrs),
                        help_text='z(t)')
    B = forms.CharField(widget=forms.Textarea(attrs=default_attrs),
                        help_text='Hyperparam B')
    x_start = forms.CharField(widget=forms.Textarea(attrs=default_attrs),
                              help_text='x_0 = S(0)')
    y_start = forms.CharField(widget=forms.Textarea(attrs=default_attrs),
                              help_text='y_0')
    steps = forms.CharField(widget=forms.NumberInput(
        attrs={'placeholder': 'Steps for t',
               'class': 'form-control'}), help_text='Steps in range')
    T = forms.CharField(widget=forms.NumberInput(
        attrs={'placeholder': 'T max for t',
               'class': 'form-control'}), help_text='Max t value')
