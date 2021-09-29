<p align="center">
	<img src="images/databiz_image.jpeg" width="200" title="logotipo_repositorio">
</p>


# Cartera

La idea principal del código es optimizar, a través de esta herramienta, la recolección efectiva  de dinero en una empresa determinada. El concepto de cartera se define en base a las deudas que tengan los clientes de una empresa. Contablemente se define como la acción activa o pasiva de recoger recursos para la empresa o entidad ya sea por sus propios recursos o mediante terceros.

Este recaudo indica a todos los ingresos(no ganancias) en determinado tiempo y se mide por la entrada de dinero(ingreso), en simples términos, las ventas realizadas en ese determinado tiempo que se realicen a la empresa

Muchos de los expertos en el tema de recaudación de cartera (recolección de dinero de la empresa) recomienda que este recaudo se recolecte y se ubique por separado según el periodo de cada uno.

---

Este código pretende responder dos preguntas fundamentales:

1. ¿Quién va a incurrir en demora en el pago (Mora)?
2. ¿Podría estimar la fecha de pago del cliente?

## Clasificación - Mora

Para hacer la predicción se utiliza la base de datos incrustada en el servidor de azure con las siguientes credenciales 

```python
server   = "carterasvr.database.windows.net"
database = "cartera"
username = "consulta"
password = "D4t4b1z2.123" 
```

esto con el fin de poder hacer la petición de la base de datos se hace necesario contactar con el administrador y registrar la ip desde la cual se va a hacer la peticion 
