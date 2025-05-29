{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
      {% for item in attributes %}
         {% if not item.startswith('_') %}
            {{ name }}.{{ item }}
         {% endif %}
      {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:

      {% if '__call__' in members %}
            {{ name }}.__call__
      {% endif %}

      {% for item in methods %}
         {% if not item.startswith('_') %}
            {{ name }}.{{ item }}
         {% endif %}
      {%- endfor %}

   {% endif %}
   {% endblock %}