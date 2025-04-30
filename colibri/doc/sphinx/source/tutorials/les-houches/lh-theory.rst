.. _lh_theory:

======
Theory
======

We follow the Les Houches parametrisation as presented in reference :cite:`Alekhin:2005xgg`.

Free parameters in the Les Houches Parametrisation
--------------------------------------------------

As described in reference :cite:`Alekhin:2005xgg`, the Les Houches parametrisation assumes that the total sea, :math:`\Sigma=u+\bar{u}+d+\bar{d}+s+\bar{s}`, is constrained to be made 40% by up and anti-up, 40% by down and anti-down, and 20% by strange and anti-strange, which means that we can write:

.. math::
    
    u+\bar{u}=0.4\Sigma,\notag\\
    d+\bar{d}=0.4\Sigma,\\
    s+\bar{s}=0.2\Sigma.\notag

It is also assumed that there is no difference between :math:`\bar{u}` and :math:`\bar{d}`, so we are only left with four active flavours, namely :math:`g, u_{v}, d_{v}` and :math:`\Sigma`. Furthermore, :math:`\epsilon_g`, :math:`\gamma_g`, :math:`\epsilon_\Sigma` and :math:`\gamma_\Sigma` are all set to zero. We are therefore left with the set of equations:

.. math::
    :label: eq:flavour-basis-set

    xf_g(x,Q_0) &= A_g\,x^{\alpha_g}\,(1 - x)^{\beta_g}, \notag \\
    xf_{u_v}(x,Q_0) &=  A_{u_v}\,x^{\alpha_{u_v}}\,(1 - x)^{\beta_{u_v}}\, (1+\epsilon_{u_v}\sqrt{x}+\gamma_{u_v} x), \\
    xf_{d_v}(x,Q_0) &= A_{d_v}\,x^{\alpha_{d_v}}\,(1 - x)^{\beta_{d_v}}\, (1+\epsilon_{d_v}\sqrt{x}+\gamma_{d_v} x),  \notag \\
    xf_\Sigma(x,Q_0) &=  A_\Sigma\,x^{\alpha_\Sigma}\,(1 - x)^{\beta_\Sigma}.\notag

This amounts to 16 parameters. Moreover, not all parameters are independent. :math:`A_g` is related to :math:`A_\Sigma` by the momentum sum rules:

.. math::
    :label: eq:gluon-sum-rule

    A_g\int_0^1 x^{\alpha_g}\,(1 - x)^{\beta_g} dx + A_\Sigma \int_0^1 x^{\alpha_\Sigma}\,(1 - x)^{\beta_\Sigma}\, dx = 1,

and the :math:`A_{u_v}` and :math:`A_{d_v}` parameters are determined by the valence sum rules:

.. math::
    :label: eq:valence-sum-rules

    & A_{u_v}\,\int\, x^{\alpha_{u_v}-1}\,(1 - x)^{\beta_{u_v}}\, (1+\epsilon_{u_v}\sqrt{x}+\gamma_{u_v} x)  dx = 2, \notag \\
    & A_{d_v}\,\int\,x^{\alpha_{d_v}-1}\,(1 - x)^{\beta_{d_v}}\, (1+\epsilon_{d_v}\sqrt{x}+\gamma_{d_v} x) dx = 1,

leaving 13 free parameters [#]_ .

.. [#] In ref. :cite:`Alekhin:2005xgg`, :math:`\epsilon_{u_v}` is fixed to its best-fit value, :math:`\epsilon_{u_v} = -1.56`, in order to avoid instability due to a very high correlation between :math:`u_v` parameters. They therefore left only 12 parameters free to vary. We decide to leave :math:`\epsilon_{u_v}` free because we don't believe we will encounter this problem.

Normalisations
--------------

In order to be able to perform a fit, we would like to write all PDFs explicitly in terms of free parameters, `x` and `Q_0`. 

We therefore write the expressions for :math:`A_g`, :math:`A_{u_v}` and :math:`A_{d_v}` explicitly by solving the integral spelled out in the sum rules, Equations :eq:`eq:gluon-sum-rule` and :eq:`eq:valence-sum-rules`, which are of the form of Euler beta functions, given by:

.. math::
    :label: eq:euler-beta-func

    \int_0^1 dt \, t^{\alpha -1} (1-t)^{\beta -1} = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)},

where, for positive integer :math:`n`, :math:`\Gamma(n)` is defined as:

.. math::

    \Gamma(n) = (n-1)!.

We find that:

.. math::

    A_g = \frac{\Gamma(\alpha_g + \beta_g + 2)}{\Gamma(\alpha_g+1)\Gamma(\beta_g+1)}\left[ 1 - A_{\Sigma} \frac{\Gamma(\alpha_\Sigma + 1) \Gamma(\beta_\Sigma + 1)}{\Gamma(\alpha_\Sigma + \beta_\Sigma +2)} \right],

.. math::

    A_{u_v} = \frac{2}{\Gamma(\beta_{u_v}+1)}\left[ \frac{\Gamma(\alpha_{u_v})}{\Gamma(\alpha_{u_v} + \beta_{u_v} + 1)}  + \epsilon_{u_v} \frac{\Gamma(\alpha_{u_v} + 1 / 2)}{\Gamma(\alpha_{u_v} + \beta_{u_v} + 3 / 2)} + \gamma_{u_v} \frac{\Gamma(\alpha_{u_v} + 1)}{\Gamma(\alpha_{u_v} + \beta_{u_v} + 2)} \right]^{-1},

.. math::

    A_{d_v} = \frac{1}{\Gamma(\beta_{d_v}+1)}\left[ \frac{\Gamma(\alpha_{d_v})}{\Gamma(\alpha_{d_v} + \beta_{d_v} + 1)}  + \epsilon_{d_v} \frac{\Gamma(\alpha_{d_v} + 1 / 2)}{\Gamma(\alpha_{d_v} + \beta_{d_v} + 3 / 2)} + \gamma_{d_v} \frac{\Gamma(\alpha_{d_v} + 1)}{\Gamma(\alpha_{d_v} + \beta_{d_v} + 2)} \right]^{-1}.


.. raw:: html

   <div class="section-title"></div>

.. _lh-evolution-basis:

The Les Houches Parametrisation in the evolution basis
------------------------------------------------------


Colibri works in the evolution basis, whose elements can be written as a linear combination of the elements of the flavour basis. 

We start by writting the elements of the evolution basis in terms of quark flavours, which is as follows:

.. math::
    :label: eq:evolution-basis

    \Sigma &= u+\bar{u}+d+\bar{d}+s+\bar{s}, \notag \\
    T_3 &= (u + \bar{u}) - (d + \bar{d}), \notag \\
    T_8 &= (u+\bar{u} + d + \bar{d}) - 2(s+\bar{s}), \\
    V &= (u-\bar{u}) + (d-\bar{d}) + (s-\bar{s}), \notag \\
    V_3 &= (u - \bar{u}) - (d - \bar{d}), \notag \\
    V_8 &= (u-\bar{u} + d-\bar{d}) - 2(s-\bar{s}). \notag

Noting that :math:`u_v = u - \bar{u}`, :math:`d_v = d - \bar{d}` and that, since there are no valence strange quarks, :math:`s_v = s - \bar{s} = 0`, and applying the assumptions stated above, we find:

.. math::
    :label: eq:flavour-basis-elements

    T_3 &= (u-\bar{d})-(d-\bar{u}) = u_v - d_v = V_3, \notag \\
    T_8 &= \Sigma - 3(s+\bar{s}) = 0.4\Sigma, \\
    V_8 &= u_v + d_v - 2 \cdot 0 = V. \notag

Therefore, we are again left with only four active flavours; :math:`\Sigma`, :math:`V`, :math:`V_3` and the gluon.

We already have an explicit parametrisation for :math:`f_\Sigma` and :math:`f_g`, as stated in Eq. :eq:`eq:flavour-basis-set`. We have the ingredients to write analogous expressions for :math:`f_V` and :math:`f_{V_3}`, which are given by:

.. math::
    :label: eq:f_V

    x f_V &= x f_{u_v} + x f_{d_v} \\
    &= A_{u_v}\,x^{\alpha_{u_v}}\,(1 - x)^{\beta_{u_v}}\, (1+\epsilon_{u_v}\sqrt{x}+\gamma_{u_v} x) + A_{d_v}\,x^{\alpha_{d_v}}\,(1 - x)^{\beta_{d_v}}(1+\epsilon_{d_v}\sqrt{x}+\gamma_{d_v} x), \notag    

.. math::
    :label: eq:f_V3

    x f_{V_3} &= x f_{u_v} - x f_{d_v} \\
    &= A_{u_v}\,x^{\alpha_{u_v}}\,(1 - x)^{\beta_{u_v}}\, (1+\epsilon_{u_v}\sqrt{x}+\gamma_{u_v} x) - A_{d_v}\,x^{\alpha_{d_v}}\,(1 - x)^{\beta_{d_v}}(1+\epsilon_{d_v}\sqrt{x}+\gamma_{d_v} x). \notag

