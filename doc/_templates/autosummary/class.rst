{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

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

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      {% for item in methods %}
         {% if item in members and (not item.startswith('_') or item in ['__call__']) %}
           {{ name }}.{{ item }}
         {% endif %}
      {%- endfor %}
   {% endif %}
   {% endblock %}