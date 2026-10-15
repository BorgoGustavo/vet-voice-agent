-- Fictitious clinic data. Names are invented and the phone numbers are placeholders.
-- Runs right after a fresh migration, so the ids below start at 1.

insert into vetvoice.vets (name) values
  ('Dra. Helena Prado'),
  ('Dr. Caio Mendes');

insert into vetvoice.clients (name, phone_e164) values
  ('Marina Lopes',     '+5511900000001'),
  ('Rafael Teixeira',  '+5511900000002'),
  ('Beatriz Nogueira', '+5511900000003');

insert into vetvoice.pets (client_id, name, species) values
  (1, 'Luna',   'cachorro'),
  (1, 'Mingau', 'gato'),
  (2, 'Thor',   'cachorro'),
  (3, 'Pipoca', 'gato');
